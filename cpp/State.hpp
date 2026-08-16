#pragma once
#include <vector>
#include <cstdint>
#include <functional>
#include <cstring>
#include <limits>
#include <cmath>
#include <unordered_map>
#include "ProblemData.hpp"

/**
 * @struct ConditionalProvenance
 * @brief Remembers that a fact went unknown because a conditional effect's
 * condition was itself unknown at apply time (ActionApplier's "Knowledge
 * Loss" rule) -- so that a LATER direct observation of that fact can be
 * abduced backward into the condition fact. Scoped to single-fact
 * (optionally negated) conditions only; see ActionApplier::apply_action.
 */
struct ConditionalProvenance {
    int condition_fact_id{-1};
    bool effect_value{false};
    bool condition_negated{false};

    bool operator==(const ConditionalProvenance& other) const {
        return condition_fact_id == other.condition_fact_id &&
               effect_value == other.effect_value &&
               condition_negated == other.condition_negated;
    }
};

/**
 * @class PartiallySpecifiedState
 * @brief Represents a single possible world using Dual-Mask bitsets.
 * Handles True, False, and Unknown states natively for contingent planning.
 */
class PartiallySpecifiedState {
public:
    std::vector<uint64_t> known_mask;
    std::vector<uint64_t> value_mask;
    std::vector<uint64_t> known_function_mask;
    std::vector<double> function_values;

    // Pending conditional-effect provenance, keyed by the effect (target)
    // fact id. See ConditionalProvenance and apply_provenance_deductions().
    std::unordered_map<int, ConditionalProvenance> pending_provenance;

    PartiallySpecifiedState() = default;

    PartiallySpecifiedState(int total_predicates, int total_functions = 0) {
        int blocks = (total_predicates / 64) + 1;
        known_mask.assign(blocks, 0ULL); 
        value_mask.assign(blocks, 0ULL);

        int func_blocks = (total_functions / 64) + 1;
        known_function_mask.assign(func_blocks, 0ULL);
        function_values.assign(total_functions, 0.0);

    }

    void set_function_value(int id, double val) {
        known_function_mask[id / 64] |= (1ULL << (id % 64));
        function_values[id] = val;
    }

    // Returns NaN if the agent does not possess the knowledge
    double get_function_value(int id) const {
        if (known_function_mask[id / 64] & (1ULL << (id % 64))) {
            return function_values[id];
        }
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Applies an arithmetic operation strictly adhering to Three Valued Logic
    void apply_numeric_effect(const NumericEffect& eff) {
        uint64_t bit = 1ULL << (eff.function_id % 64);
        bool is_known = known_function_mask[eff.function_id / 64] & bit;

        // Any arithmetic performed on an unknown value results in an unknown value
        if (!is_known && eff.op != NumericOp::ASSIGN) {
            return; 
        }

        if (eff.op == NumericOp::ASSIGN) {
            set_function_value(eff.function_id, eff.value);
        } 
        else if (eff.op == NumericOp::INCREASE) {
            function_values[eff.function_id] += eff.value;
        } 
        else if (eff.op == NumericOp::DECREASE) {
            function_values[eff.function_id] -= eff.value;
        }
    }

    bool is_true(int id) const {
        return (known_mask[id / 64] & (1ULL << (id % 64))) &&
               (value_mask[id / 64] & (1ULL << (id % 64)));
    }

    bool is_false(int id) const {
        uint64_t bit = 1ULL << (id % 64);
        return (known_mask[id / 64] & bit) && !(value_mask[id / 64] & bit);
    }

    bool is_unknown(int id) const {
        return !(known_mask[id / 64] & (1ULL << (id % 64)));
    }
    
    // Applying an observation or effect. Deliberately does NOT touch
    // pending_provenance: genuine observations go through this exact method
    // and must leave any pending entry for `id` intact so
    // apply_provenance_deductions() can still read it afterward. Forward,
    // non-observation writes that should invalidate a stale entry (an
    // unrelated action directly (re)setting this fact) call
    // clear_provenance() explicitly instead -- see ActionApplier.
    void set_known_value(int id, bool value) {
        known_mask[id / 64] |= (1ULL << (id % 64));
        if (value)
            value_mask[id / 64] |= (1ULL << (id % 64));
        else
            value_mask[id / 64] &= ~(1ULL << (id % 64));
    }

    // Explicitly discards a stale provenance entry -- used when an
    // unrelated forward mechanism (not an observation) resolves this fact,
    // making any earlier pending abduction for it moot.
    void clear_provenance(int id) {
        if (!pending_provenance.empty()) pending_provenance.erase(id);
    }

    // Equality operator masking out ignored variables dynamically
    bool operator==(const PartiallySpecifiedState& other) const {
        const std::vector<uint64_t>& comparable_mask = get_global_problem().comparable_mask;
        for (size_t i = 0; i < known_mask.size(); ++i) {
            uint64_t k1 = known_mask[i] & comparable_mask[i];
            uint64_t k2 = other.known_mask[i] & comparable_mask[i];
            if (k1 != k2) return false;

            uint64_t v1 = value_mask[i] & comparable_mask[i];
            uint64_t v2 = other.value_mask[i] & comparable_mask[i];
            if (v1 != v2) return false;
        }

        for (size_t i = 0; i < function_values.size(); ++i) {
            if (function_values[i] != other.function_values[i]) return false;
        }

        // Two bitmask-identical states can still carry different pending
        // abduction opportunities (e.g. one reached this state via a
        // knowledge-loss conditional effect, the other never triggered
        // it) -- caching them as equal would let a deduction available on
        // one path get silently lost by reusing a cached result computed
        // on the other. std::unordered_map::operator== is content-equality
        // (order-independent), so this is safe to compare directly.
        if (pending_provenance != other.pending_provenance) return false;

        return true;
    }

    // Forces a fact into the UNKNOWN state (used for Knowledge Loss and Non-Determinism)
    void set_unknown(int id) {
    uint64_t bit = 1ULL << (id % 64);

    // Canonicalize state to prevent SIMD/bitwise corruption
    // during bulk action application in later phases.
    known_mask[id / 64] &= ~bit;
    value_mask[id / 64] &= ~bit;
    // A fresh degradation supersedes any older, now-stale provenance
    // recorded for this same fact (e.g. non-determinism after an earlier
    // knowledge-loss branch, or a repeated conditional-effect hit).
    if (!pending_provenance.empty()) pending_provenance.erase(id);
}

    // Records that `effect_fact_id` would become `effect_value` if
    // `condition_fact_id` turns out (or, if `condition_negated`, turns out
    // NOT) to be true -- called by ActionApplier right after set_unknown()
    // degrades `effect_fact_id` because its conditional effect's condition
    // was itself unknown. Overwrites any prior entry for the same fact.
    void record_conditional_provenance(int effect_fact_id, int condition_fact_id, bool effect_value, bool condition_negated = false) {
        pending_provenance[effect_fact_id] = ConditionalProvenance{condition_fact_id, effect_value, condition_negated};
    }

    // Scoped regression fix: abduces a conditional effect's condition fact
    // from a LATER direct observation of its (previously knowledge-loss-
    // unknown) effect fact -- e.g. sensing `stainp(sK)` lets the engine
    // deduce `ill(iK)` in medpks-style diagnosis domains, which the
    // "Knowledge Loss" rule in ActionApplier alone can never recover.
    //
    // Deliberately one-directional: only effect-observed -> condition-
    // deduced, never the reverse (condition-becomes-known -> forward-
    // resolve effect). The condition fact may be a static hidden trait (as
    // in diagnosis domains) or a fluent a later action legitimately
    // changes, and this engine has no way to tell the two apart -- forward-
    // resolving from a *later* known condition value would silently assume
    // "static," which is unsound for the dynamic case. Call only at genuine
    // observation sites (sensing), where the just-learned fact is known to
    // reflect the world's ground truth, not an internal derivation.
    //
    // Loops to a fixpoint since resolving one condition fact can itself be
    // the effect fact of an earlier, still-pending provenance entry
    // (chained diagnosis). Cheap no-op whenever pending_provenance is empty
    // (the common case for domains without this pattern).
    bool apply_provenance_deductions() {
        if (pending_provenance.empty()) return false;
        bool any_changed = false;
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto it = pending_provenance.begin(); it != pending_provenance.end(); ) {
                int effect_fact_id = it->first;
                if (is_unknown(effect_fact_id)) {
                    ++it;
                    continue;
                }
                ConditionalProvenance prov = it->second;
                it = pending_provenance.erase(it);
                if (is_unknown(prov.condition_fact_id)) {
                    bool observed = is_true(effect_fact_id);
                    bool condition_was_true = (observed == prov.effect_value);
                    bool deduced_condition = prov.condition_negated ? !condition_was_true : condition_was_true;
                    set_known_value(prov.condition_fact_id, deduced_condition);
                }
                changed = true;
                any_changed = true;
            }
        }
        return any_changed;
    }

    // Returns false if a logical contradiction is found (dead-end state)
    bool apply_oneof_deductions(const std::vector<std::vector<int>>& oneof_groups) {
        bool state_changed = true;
        
        // Loop until epistemic equilibrium is reached (no new deductions)
        while (state_changed) {
            state_changed = false;
            
            for (const auto& group : oneof_groups) {
                int true_count = 0;
                int unknown_count = 0;
                int last_unknown_id = -1;

                // 1. Scan the current knowledge of the group
                for (int id : group) {
                    if (is_true(id)) {
                        true_count++;
                    } else if (is_unknown(id)) {
                        unknown_count++;
                        last_unknown_id = id;
                    }
                }

                // 2. Contradiction Detection
                if (true_count > 1) {
                    return false; // Mathematical impossibility, prune this state
                }

                // 3. Forward Deduction: If ONE is true, all others MUST be false
                if (true_count == 1 && unknown_count > 0) {
                    for (int id : group) {
                        if (is_unknown(id)) {
                            set_known_value(id, false);
                            state_changed = true;
                        }
                    }
                }

                // 4. Backward Deduction: If all but ONE are false, the last MUST be true
                if (true_count == 0 && unknown_count == 1) {
                    set_known_value(last_unknown_id, true);
                    state_changed = true;
                }
            }
        }
        
        return true; // Deduction complete, state is logically sound
    }
};

/**
 * @brief Utility function to securely combine hash seeds (avoids XOR cancellation).
 * Uses the 64-bit golden ratio constant.
 */
inline void hash_combine(std::size_t& seed, std::size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

/**
 * @brief Standardizes floating point bit-patterns for safe hashing.
 */
inline uint64_t canonicalize_double(double val) {
    // 1. Handle NaN variability (collapse all NaN bit-patterns into one constant)
    if (std::isnan(val)) {
        return 0x7FF8000000000000ULL; // Standard Quiet NaN bit pattern
    }
    
    // 2. Collapse negative zero to positive zero
    if (val == 0.0) { 
        val = 0.0; 
    }
    
    uint64_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    return bits;
}

/**
 * @struct StateHasher
 * @brief Cryptographically safe, collision-resistant hasher for PartiallySpecifiedState.
 */
struct StateHasher {
    std::size_t operator()(const PartiallySpecifiedState& s) const {
        std::size_t seed = 0;

        // Hash the Boolean Fluents
        const std::vector<uint64_t>& comparable_mask = get_global_problem().comparable_mask;
        for (size_t i = 0; i < s.known_mask.size(); ++i) {
            uint64_t k_block = s.known_mask[i] & comparable_mask[i];
            
            // Mask the value_block with the known_block BEFORE hashing.
            // This is a foolproof secondary safety net. Even if a garbage bit 
            // survived in value_mask, it is mathematically erased if known == 0.
            uint64_t v_block = s.value_mask[i] & k_block;

            hash_combine(seed, std::hash<uint64_t>{}(k_block));
            hash_combine(seed, std::hash<uint64_t>{}(v_block));
        }

        // Hash the Numeric Functions
        for (size_t i = 0; i < s.function_values.size(); ++i) {
            uint64_t bit = 1ULL << (i % 64);
            
            // Only hash the function value IF the agent actually knows it.
            // Hashing unknown functions causes deterministic identical states to branch.
            if (s.known_function_mask[i / 64] & bit) {
                uint64_t canonical_bits = canonicalize_double(s.function_values[i]);
                hash_combine(seed, std::hash<uint64_t>{}(canonical_bits));
            }
        }

        // Hash pending provenance order-independently (XOR per-entry hashes
        // together, since unordered_map iteration order isn't guaranteed to
        // match across two equal-content maps) -- must agree with
        // operator=='s content-equality check above. No-op when empty, so
        // this doesn't change the hash for the common case (domains without
        // this pattern never populate pending_provenance).
        std::size_t provenance_seed = 0;
        for (const auto& kv : s.pending_provenance) {
            std::size_t entry_seed = 0;
            hash_combine(entry_seed, std::hash<int>{}(kv.first));
            hash_combine(entry_seed, std::hash<int>{}(kv.second.condition_fact_id));
            hash_combine(entry_seed, std::hash<bool>{}(kv.second.effect_value));
            hash_combine(entry_seed, std::hash<bool>{}(kv.second.condition_negated));
            provenance_seed ^= entry_seed;
        }
        hash_combine(seed, provenance_seed);

        return seed;
    }
};
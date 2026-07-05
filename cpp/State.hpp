#pragma once
#include <vector>
#include <cstdint>
#include <functional>
#include <cstring>
#include <limits>
#include <cmath>
#include "ProblemData.hpp"

extern ProblemDef global_problem;

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

    PartiallySpecifiedState() = default;

    // Constructor: sizes the bitset based on Total Predicates (N)
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

    // Safely reads a function value
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

    // Mathematical State Evaluation
    bool is_true(int id) const {
        // True IF it is known AND its value is 1
        return (known_mask[id / 64] & (1ULL << (id % 64))) && 
               (value_mask[id / 64] & (1ULL << (id % 64)));
    }

    //Explicit FALSE evaluation. Must be KNOWN and VALUE == 0.
    bool is_false(int id) const {
        uint64_t bit = 1ULL << (id % 64);
        return (known_mask[id / 64] & bit) && !(value_mask[id / 64] & bit);
    }

    bool is_unknown(int id) const {
        // Unknown IF the known_mask bit is 0
        return !(known_mask[id / 64] & (1ULL << (id % 64)));
    }
    
    // Applying an observation or effect
    void set_known_value(int id, bool value) {
        known_mask[id / 64] |= (1ULL << (id % 64)); // Mark as known (1)
        if (value) 
            value_mask[id / 64] |= (1ULL << (id % 64)); // Set True
        else 
            value_mask[id / 64] &= ~(1ULL << (id % 64)); // Set False
    }

    // Equality operator masking out ignored variables dynamically
    bool operator==(const PartiallySpecifiedState& other) const {
        for (size_t i = 0; i < known_mask.size(); ++i) {
            uint64_t k1 = known_mask[i] & global_problem.comparable_mask[i];
            uint64_t k2 = other.known_mask[i] & global_problem.comparable_mask[i];
            if (k1 != k2) return false;

            uint64_t v1 = value_mask[i] & global_problem.comparable_mask[i];
            uint64_t v2 = other.value_mask[i] & global_problem.comparable_mask[i];
            if (v1 != v2) return false;
        }

        for (size_t i = 0; i < function_values.size(); ++i) {
            if (function_values[i] != other.function_values[i]) return false;
        }

        return true;
    }

    // Forces a fact into the UNKNOWN state (used for Knowledge Loss and Non-Determinism)
    void set_unknown(int id) {
    uint64_t bit = 1ULL << (id % 64);
    
    // Canonicalize state to prevent SIMD/bitwise corruption 
    // during bulk action application in later phases.
    known_mask[id / 64] &= ~bit; 
    value_mask[id / 64] &= ~bit; 
}

    // Applies SDR Logical Deduction across all OneOf constraints
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
        for (size_t i = 0; i < s.known_mask.size(); ++i) {
            uint64_t k_block = s.known_mask[i] & global_problem.comparable_mask[i];
            
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

        return seed;
    }
};
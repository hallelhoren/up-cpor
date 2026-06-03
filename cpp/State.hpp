#pragma once
#include <vector>
#include <cstdint>
#include <functional>

/**
 * @class PartiallySpecifiedState
 * @brief Represents a single possible world using Dual-Mask bitsets.
 * Handles True, False, and Unknown states natively for contingent planning.
 */
class PartiallySpecifiedState {
public:
    std::vector<uint64_t> known_mask; 
    std::vector<uint64_t> value_mask;
    std::vector<double> function_values;

    PartiallySpecifiedState() = default;

    // Constructor: sizes the bitset based on Total Predicates (N)
    PartiallySpecifiedState(int total_predicates, int total_functions) {
        int blocks = (total_predicates / 64) + 1;
        known_mask.assign(blocks, ~0ULL); 
        value_mask.assign(blocks, 0ULL);
        function_values.assign(total_functions, 0.0);

        int remainder = total_predicates % 64;
        if (remainder != 0) {
            uint64_t valid_bits_mask = (1ULL << remainder) - 1;
            known_mask.back() &= valid_bits_mask;
        }
    }

    // Mathematical State Evaluation
    bool is_true(int id) const {
        // True IF it is known AND its value is 1
        return (known_mask[id / 64] & (1ULL << (id % 64))) && 
               (value_mask[id / 64] & (1ULL << (id % 64)));
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
        known_mask[id / 64] &= ~(1ULL << (id % 64)); // Clear the known bit (0)
        // Note: The value_mask bit no longer matters because is_true checks the known_mask first
    }
};

/**
 * @struct StateHasher
 */
// High performance FNV1a hash implementation
struct StateHasher {
    std::size_t operator()(const PartiallySpecifiedState& s) const {
        std::size_t hash = 14695981039346656037ULL; // FNV offset basis

        for (size_t i = 0; i < s.known_mask.size(); ++i) {
            uint64_t k_block = s.known_mask[i] & global_problem.comparable_mask[i];
            uint64_t v_block = s.value_mask[i] & global_problem.comparable_mask[i];
            
            hash ^= k_block;
            hash *= 1099511628211ULL; // FNV prime
            
            hash ^= v_block;
            hash *= 1099511628211ULL;
        }

        for (double val : s.function_values) {
            uint64_t bits;
            std::memcpy(&bits, &val, sizeof(bits));
            hash ^= bits;
            hash *= 1099511628211ULL;
        }

        return hash;
    }
};
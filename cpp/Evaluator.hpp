#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "State.hpp"

constexpr int OP_AND = -1;
constexpr int OP_OR = -2;
constexpr int OP_NOT = -3;
constexpr int OP_ONEOF = -4; 
constexpr int OP_EQUALS = -5;
constexpr int OP_TRUE = -6;
constexpr int OP_FALSE = -7;

constexpr uint8_t VAL_FALSE = 0;
constexpr uint8_t VAL_TRUE = 1;
constexpr uint8_t VAL_UNKNOWN = 2;

// SDR Enforced Evaluation Modes
enum class EvalMode {
    PESSIMISTIC, // Demands absolute certainty (Returns True ONLY if VAL_TRUE)
    OPTIMISTIC,  // Assumes the best (Returns True if VAL_TRUE or VAL_UNKNOWN)
    EXACT        // Returns the raw 3-valued result
};

class Evaluator {
public:
    static uint8_t evaluate_rpn_raw(const std::vector<int>& rpn, const PartiallySpecifiedState& state) {
        if (rpn.empty()) return VAL_TRUE; 

        uint8_t stack[256];
        int sp = 0;
        size_t rpn_size = rpn.size();

        for (size_t i = 0; i < rpn_size; ++i) {
            int token = rpn[i];

            if (sp >= 255) throw std::runtime_error("RPN stack overflow");

            if (token >= 0) {
                if (state.is_unknown(token)) stack[sp++] = VAL_UNKNOWN;
                else if (state.is_true(token)) stack[sp++] = VAL_TRUE;
                else stack[sp++] = VAL_FALSE;
            } 
            else if (token == OP_TRUE) stack[sp++] = VAL_TRUE;
            else if (token == OP_FALSE) stack[sp++] = VAL_FALSE;
            else if (token == OP_NOT) {
                if (sp < 1) throw std::runtime_error("RPN underflow on OP NOT");
                uint8_t val = stack[sp - 1];
                if (val == VAL_TRUE) stack[sp - 1] = VAL_FALSE;
                else if (val == VAL_FALSE) stack[sp - 1] = VAL_TRUE;
            } 
            else {
                if (sp < 2) throw std::runtime_error("RPN underflow on binary OP");
                uint8_t right = stack[--sp];
                uint8_t left = stack[sp - 1];

                if (token == OP_AND) {
                    if (left == VAL_FALSE || right == VAL_FALSE) stack[sp - 1] = VAL_FALSE;
                    else if (left == VAL_TRUE && right == VAL_TRUE) stack[sp - 1] = VAL_TRUE;
                    else stack[sp - 1] = VAL_UNKNOWN;
                } 
                else if (token == OP_OR) {
                    if (left == VAL_TRUE || right == VAL_TRUE) stack[sp - 1] = VAL_TRUE;
                    else if (left == VAL_FALSE && right == VAL_FALSE) stack[sp - 1] = VAL_FALSE;
                    else stack[sp - 1] = VAL_UNKNOWN;
                }
                else if (token == OP_EQUALS) {
                    if (left == VAL_UNKNOWN || right == VAL_UNKNOWN) stack[sp - 1] = VAL_UNKNOWN;
                    else stack[sp - 1] = (left == right) ? VAL_TRUE : VAL_FALSE;
                }
                else if (token == OP_ONEOF) {
                    // Exact XOR: True if exactly one is true and the other is false.
                    if (left == VAL_UNKNOWN || right == VAL_UNKNOWN) stack[sp - 1] = VAL_UNKNOWN;
                    else stack[sp - 1] = (left != right) ? VAL_TRUE : VAL_FALSE;
                }
            }
        }
        if (sp != 1) throw std::runtime_error("RPN evaluation left multiple items on stack");
        return stack[0];
    }

    // Main interface for the Search Engine
    static bool evaluate(const std::vector<int>& rpn, const PartiallySpecifiedState& state, EvalMode mode) {
        uint8_t result = evaluate_rpn_raw(rpn, state);
        if (mode == EvalMode::PESSIMISTIC) return result == VAL_TRUE;
        if (mode == EvalMode::OPTIMISTIC) return result == VAL_TRUE || result == VAL_UNKNOWN;
        return false; // Should not reach if EXACT is handled manually
    }
};
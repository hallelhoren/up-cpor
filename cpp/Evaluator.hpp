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

// The 3 Valued Logic State Constants 
constexpr uint8_t VAL_FALSE = 0;
constexpr uint8_t VAL_TRUE = 1;
constexpr uint8_t VAL_UNKNOWN = 2;

class Evaluator {
public:
    static uint8_t evaluate_rpn(const std::vector<int>& rpn, const PartiallySpecifiedState& state) {
        if (rpn.empty()) return VAL_TRUE; 

        uint8_t stack[256];
        int sp = 0;
        size_t rpn_size = rpn.size();

        for (size_t i = 0; i < rpn_size; ++i) {
            int token = rpn[i];

            if (sp >= 255) {
                throw std::runtime_error("RPN stack overflow Formula too complex for 256 byte stack.");
            }

            if (token >= 0) {
                if (state.is_unknown(token)) {
                    stack[sp++] = VAL_UNKNOWN;
                } else {
                    stack[sp++] = state.is_true(token) ? VAL_TRUE : VAL_FALSE;
                }
            } 
            else {
                if (token == OP_TRUE) {
                    stack[sp++] = VAL_TRUE;
                }
                else if (token == OP_FALSE) {
                    stack[sp++] = VAL_FALSE;
                }
                else if (token == OP_NOT) {
                    if (sp < 1) throw std::runtime_error("RPN stack underflow on OP NOT");
                    uint8_t val = stack[sp - 1];
                    if (val == VAL_TRUE) stack[sp - 1] = VAL_FALSE;
                    else if (val == VAL_FALSE) stack[sp - 1] = VAL_TRUE;
                    else stack[sp - 1] = VAL_UNKNOWN;
                } 
                else if (token == OP_AND) {
                    if (sp < 2) throw std::runtime_error("RPN stack underflow on OP AND");
                    uint8_t right = stack[--sp];
                    uint8_t left = stack[sp - 1];

                    if (left == VAL_FALSE || right == VAL_FALSE) stack[sp - 1] = VAL_FALSE;
                    else if (left == VAL_TRUE && right == VAL_TRUE) stack[sp - 1] = VAL_TRUE;
                    else stack[sp - 1] = VAL_UNKNOWN;
                } 
                else if (token == OP_OR) {
                    if (sp < 2) throw std::runtime_error("RPN stack underflow on OP OR");
                    uint8_t right = stack[--sp];
                    uint8_t left = stack[sp - 1];

                    if (left == VAL_TRUE || right == VAL_TRUE) stack[sp - 1] = VAL_TRUE;
                    else if (left == VAL_FALSE && right == VAL_FALSE) stack[sp - 1] = VAL_FALSE;
                    else stack[sp - 1] = VAL_UNKNOWN;
                }
                else if (token == OP_EQUALS) {
                    if (sp < 2) throw std::runtime_error("RPN stack underflow on OP EQUALS");
                    uint8_t right = stack[--sp];
                    uint8_t left = stack[sp - 1];
                    
                    if (left == VAL_UNKNOWN || right == VAL_UNKNOWN) stack[sp - 1] = VAL_UNKNOWN;
                    else stack[sp - 1] = (left == right ? VAL_TRUE : VAL_FALSE);
                }
            }
        }
        
        if (sp < 1) throw std::runtime_error("RPN evaluation failed empty stack");
        return stack[0];
    }
};
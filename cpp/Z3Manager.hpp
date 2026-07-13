#pragma once
#include <z3++.h>
#include <vector>
#include <string>
#include <stdexcept>
#include "ProblemData.hpp"
#include "Evaluator.hpp"

namespace CPOR {

class Z3Manager {
public:
    z3::context ctx;
    z3::solver solver;
    std::vector<z3::expr> fluent_vars;
    bool is_initialized;

    Z3Manager() : solver(ctx), is_initialized(false) {}

    // Translates a flattened RPN formula (the same contract Evaluator::evaluate_rpn_raw
    // consumes) into a live Z3 boolean expression over the persistent fluent variables.
    // Shares the opcode set defined in Evaluator.hpp so the two evaluators never drift.
    z3::expr rpn_to_z3(const std::vector<int>& rpn) {
        std::vector<z3::expr> stack;
        stack.reserve(rpn.size());

        for (int token : rpn) {
            if (token >= 0) {
                stack.push_back(fluent_vars[token]);
            } else if (token == OP_TRUE) {
                stack.push_back(ctx.bool_val(true));
            } else if (token == OP_FALSE) {
                stack.push_back(ctx.bool_val(false));
            } else if (token == OP_NOT) {
                if (stack.empty()) throw std::runtime_error("RPN underflow on OP_NOT during Z3 translation");
                z3::expr a = stack.back(); stack.pop_back();
                stack.push_back(!a);
            } else {
                if (stack.size() < 2) throw std::runtime_error("RPN underflow on binary op during Z3 translation");
                z3::expr right = stack.back(); stack.pop_back();
                z3::expr left = stack.back(); stack.pop_back();

                if (token == OP_AND) stack.push_back(left && right);
                else if (token == OP_OR) stack.push_back(left || right);
                else if (token == OP_EQUALS) stack.push_back(left == right);
                else throw std::runtime_error("Unsupported RPN opcode during Z3 translation");
            }
        }

        if (stack.size() != 1) throw std::runtime_error("Malformed RPN passed to Z3 translation");
        return stack.back();
    }

    // Discards any state built for a previously-loaded problem. Must be called
    // whenever a new problem is loaded into a thread that may have already
    // solved a different problem -- fluent_vars, the permanently-asserted
    // dead-end constraints, and problem.oneofs itself (consulted per-query by
    // assert_relevant_oneofs) are only valid for the problem they were built
    // from, and reusing them for a problem with a different total_predicates
    // (or different oneofs/deadends) is undefined behavior (out-of-bounds
    // fluent_vars[] access) or silent cross-problem constraint corruption.
    void invalidate() {
        if (!is_initialized) return;
        solver.reset();
        fluent_vars.clear();
        is_initialized = false;
    }

    // Initializes the context once per problem (call invalidate() first if a
    // different problem was previously loaded in this thread).
    void initialize(const ProblemDef& problem) {
        if (is_initialized) return;

        fluent_vars.reserve(problem.total_predicates);

        // 1. Create persistent Z3 Boolean Variables
        for (int i = 0; i < problem.total_predicates; ++i) {
            std::string var_name = "f_" + std::to_string(i);
            fluent_vars.push_back(ctx.bool_const(var_name.c_str()));
        }

        // 2. OneOf groups are intentionally NOT asserted here anymore -- see
        // assert_relevant_oneofs() below for why and where they're asserted instead.

        // 3. Assert Dead-End Formulas permanently at depth 0: a witness state is only
        // mathematically valid if it does NOT satisfy any known dead-end condition.
        // Without this, sampled witnesses could land in states the engine already
        // knows are unsolvable, corrupting the heuristic and wasting search effort.
        for (const auto& deadend_rpn : problem.deadend_rpns) {
            if (deadend_rpn.empty()) continue;
            solver.add(!rpn_to_z3(deadend_rpn));
        }

        is_initialized = true;
    }

    // Asserts each oneof group's "exactly one member is true" invariant, but only
    // for groups still consistent with `belief`'s already-KNOWN facts -- i.e. skips
    // a group entirely if the belief state already shows it's moot (every member
    // known false, meaning whatever object/fact the group tracked has since moved
    // outside the group entirely -- e.g. unix's `mv` relocating a file from one of
    // its originally-uncertain candidate directories to a directory outside that
    // set) or already contradictory (more than one member known true).
    //
    // This must be called from inside the caller's solver.push()/pop() scope (see
    // SDRSampler::sample_concrete_states), not from initialize(): a oneof group was
    // previously asserted permanently at depth 0, which was sound only as long as
    // the fact set it described never changed. Once an action's effects move a
    // tracked fact outside its oneof group, the group's invariant becomes stale for
    // every state reached from that point on -- but a permanent, depth-0 assertion
    // has no way to "expire", so it kept contradicting the (now completely valid)
    // resulting belief state, `solver.check()` returned unsat, and the heuristic
    // reported a mathematically dead state for a state that was actually a step
    // away from the goal. Deciding relevance per-query and re-asserting only the
    // still-applicable groups inside a push()/pop() scope means a stale group is
    // silently dropped for exactly the queries it no longer applies to, while
    // remaining fully enforced for every query where it's still meaningful (in
    // particular the initial belief state, where every group is always relevant).
    void assert_relevant_oneofs(const PartiallySpecifiedState& belief, const ProblemDef& problem) {
        for (const auto& oneof_group : problem.oneofs) {
            if (oneof_group.empty()) continue;

            int known_true_count = 0;
            int known_false_count = 0;
            for (int fluent_id : oneof_group) {
                if (belief.is_true(fluent_id)) known_true_count++;
                else if (belief.is_false(fluent_id)) known_false_count++;
            }

            // Stale (tracked fact moved outside the group -- every member already
            // known false) or already contradictory (shouldn't happen if
            // apply_oneof_deductions ran correctly, but never assert something that
            // would make an otherwise-valid state spuriously unsat). Either way,
            // this group carries no useful information for this specific query.
            if (known_false_count == static_cast<int>(oneof_group.size()) || known_true_count > 1) {
                continue;
            }

            z3::expr_vector group_vars(ctx);
            for (int fluent_id : oneof_group) {
                group_vars.push_back(fluent_vars[fluent_id]);
            }

            // At least one is true
            solver.add(z3::mk_or(group_vars));

            // Mutually exclusive (at most one is true)
            for (size_t i = 0; i < oneof_group.size(); ++i) {
                for (size_t j = i + 1; j < oneof_group.size(); ++j) {
                    solver.add(!(fluent_vars[oneof_group[i]] && fluent_vars[oneof_group[j]]));
                }
            }
        }
    }
};

} // namespace CPOR
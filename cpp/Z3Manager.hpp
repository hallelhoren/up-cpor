#pragma once
#include <z3++.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include "ProblemData.hpp"
#include "Evaluator.hpp"
#include "DebugStats.hpp"

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
    //
    // assert_deadends controls whether known dead-end formulas get baked in
    // as permanent depth-0 axioms (see step 3 below) -- true for the witness
    // -sampling use case this was originally built for, but it must be false
    // for a Z3Manager instance used for general entailment queries (see
    // formula_is_entailed): baking in "this dead-end formula is impossible"
    // as an axiom makes any later query asking "is this dead-end formula
    // false, given the CURRENT belief" a tautology -- true regardless of
    // what the belief actually says -- which is exactly backwards for a
    // caller trying to determine whether a *specific* belief has actually
    // ruled a dead-end out. See g_z3_entailment_manager's own comment.
    void initialize(const ProblemDef& problem, bool assert_deadends = true) {
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
        if (assert_deadends) {
            for (const auto& deadend_rpn : problem.deadend_rpns) {
                if (deadend_rpn.empty()) continue;
                solver.add(!rpn_to_z3(deadend_rpn));
            }
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

    // Checks whether `rpn` is logically ENTAILED by `belief`'s known facts
    // together with the oneof groups still relevant to `belief` (see
    // assert_relevant_oneofs) -- i.e. whether belief + NOT(rpn) is
    // UNSATISFIABLE. Deliberately does NOT bake in the problem's dead-end
    // formulas as axioms the way sample_concrete_states does -- see this
    // method's initialize() call and g_z3_entailment_manager's own comment
    // for why that would make dead-end-adjacent queries vacuous.
    //
    // This is a strictly more capable alternative to Evaluator::evaluate for
    // formulas whose truth only follows once a domain invariant is factored
    // in: a flat, invariant-blind three-valued evaluation has no notion of
    // "these five room-identity facts are mutually exclusive," so it can
    // never resolve a regressed disjunction like the one
    // RegressionEngine::regress_rpn now builds when a token is gated by
    // several still-unresolved conditional effects (e.g. a localization
    // domain's per-room `checking` conditions) -- Z3 can, because the same
    // oneof invariant that already drives witness sampling is asserted here
    // too. Uses the same push()/assert/check()/pop() scoping already proven
    // safe by sample_concrete_states, so a query's assertions never leak
    // into the next one.
    // extra_constraints (default empty): additional RPN formulas asserted
    // alongside `belief`'s own known facts -- see
    // BeliefState::verify_condition_safely for why this matters: `rpn` alone
    // only captures what regressing the TARGET query implies, which is a
    // no-op for any token an observation never directly modified (e.g. a
    // hidden identity fact like `at(room)` that only sensing actions'
    // *effects on other facts* constrain, never write to directly). What was
    // actually LEARNED from each observation along the way has to be
    // regressed to the initial state in its own right and fed in here, or
    // it never influences this check at all.
    bool formula_is_entailed(const std::vector<int>& rpn, const PartiallySpecifiedState& belief, const ProblemDef& problem,
                              const std::vector<std::vector<int>>& extra_constraints = {}) {
        if (rpn.empty()) return true;
        // false: this instance must NOT carry the "known dead-ends are
        // impossible" axioms sample_concrete_states relies on -- see this
        // method's own initialize() overload comment for why baking those in
        // here would make dead-end-adjacent entailment queries vacuous.
        initialize(problem, /*assert_deadends=*/false);

        solver.push();
        assert_relevant_oneofs(belief, problem);
        for (int i = 0; i < problem.total_predicates; ++i) {
            if (!belief.is_unknown(i)) {
                if (belief.is_true(i)) solver.add(fluent_vars[i]);
                else solver.add(!fluent_vars[i]);
            }
        }
        for (const auto& constraint_rpn : extra_constraints) {
            if (!constraint_rpn.empty()) solver.add(rpn_to_z3(constraint_rpn));
        }
        solver.add(!rpn_to_z3(rpn));
        auto __dbg_t0 = DebugStats::enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        z3::check_result result = solver.check();
        if (DebugStats::enabled()) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - __dbg_t0).count();
            DebugStats::record_z3_call(ms);
        }
        solver.pop();

        // UNSAT means belief's known facts + invariants are logically
        // incompatible with the formula being false -- i.e. the formula must
        // be true. SAT (a counterexample exists) or UNKNOWN (Z3 gave up) both
        // mean entailment isn't proven; treated the same, conservatively, as
        // "can't conclude true" -- never a false positive.
        return result == z3::unsat;
    }
};

// Shared, thread-local Z3 context/solver for witness sampling
// (SDRSampler::sample_concrete_states): lock-free (each thread owns its own
// instance), initialized WITH the known-dead-ends-are-impossible axioms
// (initialize()'s default assert_deadends=true) -- exactly what sampling
// needs so it never proposes a witness that's already a confirmed dead end.
inline thread_local Z3Manager g_z3_manager;

// Separate thread-local instance for general logical-entailment queries
// (BeliefState::verify_condition_safely's Z3 fallback -- see
// formula_is_entailed). Deliberately NOT the same instance as g_z3_manager:
// that one bakes dead-end formulas in as permanent axioms, which is correct
// for filtering sample witnesses but would make any entailment query
// touching a dead-end formula (e.g. "is this dead-end false, given the
// current belief") vacuously true regardless of the belief -- exactly
// backwards for a caller trying to determine whether a *specific* belief has
// actually ruled a dead-end out. Kept as its own instance (rather than a
// runtime toggle on g_z3_manager) so a query on one can never accidentally
// observe axioms baked in by the other, whichever gets initialized first.
inline thread_local Z3Manager g_z3_entailment_manager;

} // namespace CPOR
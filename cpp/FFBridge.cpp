#include "FFBridge.hpp"
#include "Evaluator.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <utility>

// Real C ABI, matching FF-v2.3/ff_api.h exactly.
extern "C" {
    int ff_load_problem(
        int num_facts,
        int num_ops,
        int num_effects,
        const int* effect_op,
        const int* effect_pc_flat, const int* effect_pc_offsets, const int* effect_pc_counts,
        const int* effect_add_flat, const int* effect_add_offsets, const int* effect_add_counts,
        const int* effect_del_flat, const int* effect_del_offsets, const int* effect_del_counts,
        int num_goal_facts, const int* goal_facts
    );
    int* ff_search(const uint64_t* determinized_state_bitset, int* out_plan_length);
    int ff_estimate_heuristic(const uint64_t* determinized_state_bitset,
                               int* out_helpful_actions, int max_helpful_actions,
                               int* out_num_helpful);
    void ff_reset_search_state();
    void ff_clear_hash_table();
}

namespace CPOR {

bool FFBridge::s_available = false;
int FFBridge::s_total_predicates = 0;
std::vector<int> FFBridge::s_shadow_fact_of;
int FFBridge::s_num_ff_facts = 0;
int FFBridge::s_num_ops = 0;

namespace {

// A literal is (predicate_id, required_value): required_value == true means
// the predicate must hold, false means its negation must hold.
using Literal = std::pair<int, bool>;

enum class RpnKind { Unrepresentable, Constant, Conjunction };

struct RpnResult {
    RpnKind kind;
    bool const_value = false;
    std::vector<Literal> literals; // deduplicated by predicate id when kind == Conjunction

    static RpnResult unrepresentable() { return RpnResult{RpnKind::Unrepresentable, false, {}}; }
    static RpnResult constant(bool v) { return RpnResult{RpnKind::Constant, v, {}}; }
    static RpnResult conjunction(std::vector<Literal> lits) {
        return RpnResult{RpnKind::Conjunction, false, std::move(lits)};
    }
};

void add_literal(std::vector<Literal>& lits, Literal lit) {
    for (const auto& existing : lits) {
        if (existing == lit) return;
    }
    lits.push_back(lit);
}

RpnResult combine_and(const RpnResult& a, const RpnResult& b) {
    // Relaxation: an unrepresentable (genuinely disjunctive) conjunct is
    // dropped -- treated as vacuously true -- instead of poisoning the whole
    // conjunction. FFBridge's output is only ever consumed as a heuristic
    // score or an OnlinePlan proposal that gets independently re-validated
    // against the real belief before anything is committed to (see this
    // file's class-level doc comment), so over-approximating a precondition
    // this way can only make FF occasionally suggest something inapplicable
    // -- already an expected, handled case -- never silently accept an
    // actually-invalid plan as ground truth.
    if (a.kind == RpnKind::Unrepresentable) return b;
    if (b.kind == RpnKind::Unrepresentable) return a;
    if (a.kind == RpnKind::Constant && !a.const_value) return RpnResult::constant(false);
    if (b.kind == RpnKind::Constant && !b.const_value) return RpnResult::constant(false);
    if (a.kind == RpnKind::Constant && a.const_value) return b;
    if (b.kind == RpnKind::Constant && b.const_value) return a;

    // Both are conjunctions.
    std::vector<Literal> merged = a.literals;
    for (const auto& lit : b.literals) add_literal(merged, lit);
    return RpnResult::conjunction(std::move(merged));
}

RpnResult combine_or(const RpnResult& a, const RpnResult& b) {
    if (a.kind == RpnKind::Constant && a.const_value) return RpnResult::constant(true);
    if (b.kind == RpnKind::Constant && b.const_value) return RpnResult::constant(true);
    if (a.kind == RpnKind::Constant && !a.const_value) return b;
    if (b.kind == RpnKind::Constant && !b.const_value) return a;
    // Neither side is a degenerate constant: a genuine disjunction, which a
    // single conjunction of literals cannot represent.
    return RpnResult::unrepresentable();
}

RpnResult combine_not(const RpnResult& a) {
    if (a.kind == RpnKind::Unrepresentable) return RpnResult::unrepresentable();
    if (a.kind == RpnKind::Constant) return RpnResult::constant(!a.const_value);
    // NOT of a genuine (multi-literal or empty) conjunction is a disjunction
    // via De Morgan -- only representable when it's a single literal, where
    // negation is just a flip.
    if (a.literals.size() == 1) {
        return RpnResult::conjunction({{a.literals[0].first, !a.literals[0].second}});
    }
    return RpnResult::unrepresentable();
}

RpnResult combine_equals(const RpnResult& a, const RpnResult& b) {
    // EQUALS(a,b) = (a AND b) OR (NOT a AND NOT b) -- inherently disjunctive
    // unless one side collapses to a constant, in which case EQUALS
    // degenerates into an identity or a negation.
    if (a.kind == RpnKind::Constant) return a.const_value ? b : combine_not(b);
    if (b.kind == RpnKind::Constant) return b.const_value ? a : combine_not(a);
    return RpnResult::unrepresentable();
}

// Mirrors Evaluator::evaluate_rpn_raw's stack-based dispatch exactly, over
// RpnResult instead of a three-valued truth value.
RpnResult evaluate_rpn_symbolic(const std::vector<int>& rpn) {
    if (rpn.empty()) return RpnResult::constant(true);

    std::vector<RpnResult> stack;
    stack.reserve(rpn.size());

    for (int token : rpn) {
        if (token >= 0) {
            stack.push_back(RpnResult::conjunction({{token, true}}));
        } else if (token == OP_TRUE) {
            stack.push_back(RpnResult::constant(true));
        } else if (token == OP_FALSE) {
            stack.push_back(RpnResult::constant(false));
        } else if (token == OP_NOT) {
            if (stack.empty()) return RpnResult::unrepresentable();
            RpnResult a = std::move(stack.back());
            stack.pop_back();
            stack.push_back(combine_not(a));
        } else {
            if (stack.size() < 2) return RpnResult::unrepresentable();
            RpnResult right = std::move(stack.back());
            stack.pop_back();
            RpnResult left = std::move(stack.back());
            stack.pop_back();

            if (token == OP_AND) stack.push_back(combine_and(left, right));
            else if (token == OP_OR) stack.push_back(combine_or(left, right));
            else if (token == OP_EQUALS) stack.push_back(combine_equals(left, right));
            else return RpnResult::unrepresentable();
        }
    }

    if (stack.size() != 1) return RpnResult::unrepresentable();
    return stack.back();
}

// One fully-analyzed effect, ready to be lowered to an FF EfConn once shadow
// fact ids are known. `condition` is the effect's OWN condition only (not yet
// unioned with the operator's precondition -- that happens during lowering,
// once every operator's precondition has also been confirmed representable).
struct AnalyzedEffect {
    std::vector<Literal> condition; // conjunction of literals gating this effect
    std::vector<Literal> effects;   // (fact_id, new_value) pairs this effect applies
};

struct AnalyzedOperator {
    std::vector<Literal> precondition;
    std::vector<AnalyzedEffect> effects;
};

} // namespace

bool FFBridge::build(const ProblemDef& problem) {
    s_available = false;
    s_total_predicates = problem.total_predicates;
    s_shadow_fact_of.assign(problem.total_predicates, -1);
    s_num_ff_facts = 0;

    // --- Pass 1: analyze every formula in the problem. Any failure aborts
    // the whole build -- a partially-translated problem would let FF silently
    // understate reachability for the actions/effects it couldn't represent. ---
    RpnResult goal_result = evaluate_rpn_symbolic(problem.goal_rpn);
    if (goal_result.kind == RpnKind::Unrepresentable) return false;
    if (goal_result.kind == RpnKind::Constant && !goal_result.const_value) {
        // Goal is unconditionally false -- a degenerate case better left to
        // the bounded-BFS fallback (which handles it via its own goal check)
        // than special-cased here.
        return false;
    }

    std::vector<AnalyzedOperator> ops;
    ops.reserve(problem.actions.size());

    for (const auto& action : problem.actions) {
        RpnResult pre = evaluate_rpn_symbolic(action.precondition_rpn);

        AnalyzedOperator op;
        if (pre.kind == RpnKind::Conjunction) {
            op.precondition = pre.literals;
        } else if (pre.kind == RpnKind::Constant && !pre.const_value) {
            // Never applicable: keep the operator slot (so effect_op indices
            // stay simple 1:1 with problem.actions) but give it zero effects.
            ops.push_back(std::move(op));
            continue;
        }
        // pre.kind == Constant(true), or Unrepresentable (e.g. a precondition
        // that's ENTIRELY a structural disjunction with no representable
        // literal conjunct to fall back on, such as navigate-to's bare
        // `(forall (?c) (or (not (inside ?o ?c)) (open ?c)))`, where
        // combine_and's own per-conjunct relaxation above has nothing left
        // to combine with): op.precondition stays empty (always-true in FF's
        // model) rather than aborting this action's translation, let alone
        // the whole problem's -- see this file's class-level doc comment.

        AnalyzedEffect guaranteed;
        guaranteed.condition = {}; // gated only by the operator's own precondition
        for (const auto& eff : action.guaranteed_effects) {
            guaranteed.effects.emplace_back(eff.first, eff.second);
        }
        for (int nd_fact : action.non_deterministic_effects) {
            // Conservative, sound-for-reachability approximation: a
            // non-deterministic effect might make the fact true, so treat it
            // as a (relaxed) add. Never treated as a delete, and its shadow
            // is left untouched -- this can only ever make FF's estimate of
            // reachability more optimistic for this fact, never falsely
            // prune a genuinely reachable action.
            guaranteed.effects.emplace_back(nd_fact, true);
        }
        if (action.observe_predicate_id != -1) {
            // Sensing itself has no direct effect on the world -- but for FF's
            // reachability graph, credit it exactly the way a non-deterministic
            // effect is credited just above: an optimistic relaxed add of the
            // observed fact, at real unit action cost (ff_search's plan length
            // already counts every operator, including this one, as one step).
            // Without this, a pure :observe action (no guaranteed/conditional/
            // non-deterministic effects at all) is completely invisible to FF --
            // any plan that genuinely needs to sense before some other action's
            // precondition can fire looks permanently unreachable. That's
            // exactly the "heuristic is blind to the epistemic value of
            // sensing" bug confirmed on localize5-tamer: 0 successful subtrees
            // found across 300k+ search-node visits before this fix.
            guaranteed.effects.emplace_back(action.observe_predicate_id, true);
        }
        if (!guaranteed.effects.empty() || action.guaranteed_effects.empty()) {
            // Always register at least one effect (even a no-op one) so the
            // operator's own precondition is registered and it can be
            // recognized as applicable by FF's relaxed reachability.
            op.effects.push_back(std::move(guaranteed));
        }

        for (const auto& ce : action.conditional_effects) {
            RpnResult cond = evaluate_rpn_symbolic(ce.condition_rpn);
            if (cond.kind == RpnKind::Unrepresentable) {
                // Relaxation: same reasoning as the per-action precondition
                // case above, but conservatively resolved the opposite way --
                // treat an unrepresentable condition as never firing (rather
                // than always firing) so FF doesn't start believing in facts
                // this effect wouldn't actually establish in most real
                // states, which would corrupt its causal reachability model
                // more than simply not crediting this one effect does.
                continue;
            }
            if (cond.kind == RpnKind::Constant && !cond.const_value) {
                continue; // Never fires; sound to omit entirely.
            }

            AnalyzedEffect ae;
            if (cond.kind == RpnKind::Conjunction) ae.condition = cond.literals;
            // Constant(true): ae.condition stays empty, correct.
            for (const auto& eff : ce.effects) {
                ae.effects.emplace_back(eff.first, eff.second);
            }
            op.effects.push_back(std::move(ae));
        }

        ops.push_back(std::move(op));
    }

    // --- Pass 2: assign shadow "not-P" facts for every predicate referenced
    // as a negative literal anywhere (preconditions, effect conditions, and
    // the goal). ---
    auto scan_for_negatives = [&](const std::vector<Literal>& lits) {
        for (const auto& lit : lits) {
            if (!lit.second && s_shadow_fact_of[lit.first] == -1) {
                s_shadow_fact_of[lit.first] = 0; // placeholder, assigned below
            }
        }
    };
    scan_for_negatives(goal_result.kind == RpnKind::Conjunction ? goal_result.literals
                                                                 : std::vector<Literal>{});
    for (const auto& op : ops) {
        scan_for_negatives(op.precondition);
        for (const auto& eff : op.effects) scan_for_negatives(eff.condition);
    }

    int next_fact_id = problem.total_predicates;
    for (int i = 0; i < problem.total_predicates; ++i) {
        if (s_shadow_fact_of[i] == 0) {
            s_shadow_fact_of[i] = next_fact_id++;
        }
    }
    s_num_ff_facts = next_fact_id;

    // --- Pass 3: lower to FF's flat C arrays. ---
    auto literal_to_ff_fact = [&](const Literal& lit) -> int {
        if (lit.second) return lit.first;
        // Negative literal: must have a shadow by construction (assigned above).
        return s_shadow_fact_of[lit.first];
    };

    std::vector<int> effect_op;
    std::vector<int> effect_pc_flat, effect_pc_offsets, effect_pc_counts;
    std::vector<int> effect_add_flat, effect_add_offsets, effect_add_counts;
    std::vector<int> effect_del_flat, effect_del_offsets, effect_del_counts;

    for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
        const AnalyzedOperator& op = ops[op_idx];
        for (const auto& eff : op.effects) {
            effect_op.push_back(static_cast<int>(op_idx));

            // PC = operator precondition UNION this effect's own condition.
            std::vector<int> pc_facts;
            pc_facts.reserve(op.precondition.size() + eff.condition.size());
            for (const auto& lit : op.precondition) pc_facts.push_back(literal_to_ff_fact(lit));
            for (const auto& lit : eff.condition) pc_facts.push_back(literal_to_ff_fact(lit));
            std::sort(pc_facts.begin(), pc_facts.end());
            pc_facts.erase(std::unique(pc_facts.begin(), pc_facts.end()), pc_facts.end());

            effect_pc_offsets.push_back(static_cast<int>(effect_pc_flat.size()));
            effect_pc_counts.push_back(static_cast<int>(pc_facts.size()));
            effect_pc_flat.insert(effect_pc_flat.end(), pc_facts.begin(), pc_facts.end());

            // Add/delete lists, with shadow mirroring so "not-P" stays consistent.
            std::vector<int> add_facts, del_facts;
            for (const auto& fx : eff.effects) {
                int predicate_id = fx.first;
                bool becomes_true = fx.second;
                if (becomes_true) {
                    add_facts.push_back(predicate_id);
                    if (s_shadow_fact_of[predicate_id] != -1) del_facts.push_back(s_shadow_fact_of[predicate_id]);
                } else {
                    del_facts.push_back(predicate_id);
                    if (s_shadow_fact_of[predicate_id] != -1) add_facts.push_back(s_shadow_fact_of[predicate_id]);
                }
            }

            effect_add_offsets.push_back(static_cast<int>(effect_add_flat.size()));
            effect_add_counts.push_back(static_cast<int>(add_facts.size()));
            effect_add_flat.insert(effect_add_flat.end(), add_facts.begin(), add_facts.end());

            effect_del_offsets.push_back(static_cast<int>(effect_del_flat.size()));
            effect_del_counts.push_back(static_cast<int>(del_facts.size()));
            effect_del_flat.insert(effect_del_flat.end(), del_facts.begin(), del_facts.end());
        }
    }

    std::vector<int> goal_facts;
    if (goal_result.kind == RpnKind::Conjunction) {
        for (const auto& lit : goal_result.literals) goal_facts.push_back(literal_to_ff_fact(lit));
    }
    // Constant(true) goal: goal_facts stays empty, correctly "always satisfied".

    s_num_ops = static_cast<int>(ops.size());

    int ok = ff_load_problem(
        s_num_ff_facts,
        static_cast<int>(ops.size()),
        static_cast<int>(effect_op.size()),
        effect_op.empty() ? nullptr : effect_op.data(),
        effect_pc_flat.empty() ? nullptr : effect_pc_flat.data(),
        effect_pc_offsets.empty() ? nullptr : effect_pc_offsets.data(),
        effect_pc_counts.empty() ? nullptr : effect_pc_counts.data(),
        effect_add_flat.empty() ? nullptr : effect_add_flat.data(),
        effect_add_offsets.empty() ? nullptr : effect_add_offsets.data(),
        effect_add_counts.empty() ? nullptr : effect_add_counts.data(),
        effect_del_flat.empty() ? nullptr : effect_del_flat.data(),
        effect_del_offsets.empty() ? nullptr : effect_del_offsets.data(),
        effect_del_counts.empty() ? nullptr : effect_del_counts.data(),
        static_cast<int>(goal_facts.size()),
        goal_facts.empty() ? nullptr : goal_facts.data()
    );

    // Clears FF's longer-lived state-hash table -- distinct from
    // ff_reset_search_state() (called at the top of every search() call, for
    // per-search EHC/BFS progress flags only). Without this, a hash table
    // that persists across every search() call within one problem's lifetime
    // was never cleared BETWEEN problems either: a process that solves more
    // than one problem in sequence (any test run exercising multiple domains,
    // not just this specific one) reused stale entries keyed by a PREVIOUS
    // problem's own fact/op numbering against a brand new problem's
    // numbering, corrupting FF's search for every problem after the first.
    // Confirmed via direct reproduction: localize5-tamer converges soundly in
    // under 3 seconds run alone, but times out after 60+ seconds when even
    // ONE other domain's problem was solved earlier in the same process --
    // narrowed to this exact missing call (ff_clear_hash_table was declared
    // in the extern "C" block above but never invoked anywhere).
    ff_clear_hash_table();

    s_available = (ok != 0);
    return s_available;
}

bool FFBridge::is_available() {
    return s_available;
}

std::vector<uint64_t> FFBridge::build_extended_bitset(const PartiallySpecifiedState& concrete_state) {
    std::vector<uint64_t> extended((static_cast<size_t>(s_num_ff_facts) / 64) + 1, 0ULL);
    for (int i = 0; i < s_total_predicates; ++i) {
        bool val = concrete_state.is_true(i);
        if (val) {
            extended[i / 64] |= (1ULL << (i % 64));
        } else if (s_shadow_fact_of[i] != -1) {
            int shadow = s_shadow_fact_of[i];
            extended[shadow / 64] |= (1ULL << (shadow % 64));
        }
    }
    return extended;
}

std::vector<int> FFBridge::search(const PartiallySpecifiedState& concrete_state) {
    if (!s_available) return {};

    // Required before every search turn: clears FF's EHC/BFS hash tables and
    // per-fact/per-op search-progress flags left over from the previous call.
    ff_reset_search_state();

    std::vector<uint64_t> extended = build_extended_bitset(concrete_state);

    int plan_length = 0;
    int* raw_plan = ff_search(extended.data(), &plan_length);
    if (raw_plan == nullptr || plan_length < 0) {
        return {};
    }

    std::vector<int> plan(raw_plan, raw_plan + plan_length);
    std::free(raw_plan);
    return plan;
}

int FFBridge::estimate_heuristic(const PartiallySpecifiedState& concrete_state,
                                   std::vector<int>& out_helpful_action_ids) {
    out_helpful_action_ids.clear();
    if (!s_available) return 999999;

    // Deliberately NOT ff_reset_search_state(): get_1P_and_H is self-contained
    // and doesn't touch the EHC/BFS hash tables that call resets -- see
    // ff_api.h's doc comment on ff_estimate_heuristic for why skipping this
    // (a real, measured cost on every call) is safe here.
    std::vector<uint64_t> extended = build_extended_bitset(concrete_state);

    // Sized to s_num_ops (== problem.actions.size() at the most recent
    // build(), see its own doc comment) so ff_estimate_heuristic never
    // truncates gH -- gnum_H can't exceed FF's own loaded operator count.
    std::vector<int> helpful(static_cast<size_t>(s_num_ops));
    int num_helpful = 0;
    int h = ff_estimate_heuristic(extended.data(), helpful.data(), s_num_ops, &num_helpful);
    if (h < 0) return 999999; // FF's own INFINITY sentinel

    helpful.resize(static_cast<size_t>(num_helpful));
    out_helpful_action_ids = std::move(helpful);
    return h;
}

} // namespace CPOR

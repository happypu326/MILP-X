"""
Constraint-side utilities for the "Constraint Matters" model-reduction method.

Instead of (only) predicting variable values, this method predicts *critical
tight constraints* (CTCs) -- inequalities that are tight (hold with equality) at
the optimum and whose type is informative -- and, at solve time, softly turns a
chosen subset into equalities to shrink the feasible region.

This module provides the dependency-light pieces used to build labels and to
reconstruct constraint rows:

  * get_sorted_constraints(ins_path): the instance's linear constraints in the
    SAME order get_a_new2 uses (sorted by (#nonzeros, str(constraint))), each as
    (name, coeff_dict, rhs, sense) with sense in {0: '<=', 1: '>=', 2: '=='}.
    This guarantees per-constraint predictions/labels align with the bipartite
    graph's constraint nodes (row i for i < ncons; the appended objective node
    is row ncons and is never a CTC).
  * classify_constraint / TYPE_RHO: a light MIPLIB-prototype classifier and an
    information-theoretic priority (lower rho => higher info gain -log rho).
  * compute_ctc_labels(...): tight mask at a solution, optionally restricted to
    the high-priority types selected by the TCP "JUDGE" rule.

Reference: "Constraint Matters: Multi-Modal Representation for Reducing MILP"
(ICLR 2026). This implements the dependency-light core (no T5 abstract graph);
the paper's own ablation shows the constraint-reduction core already improves
over variable-only PS.
"""

import numpy as np
import pyscipopt as scp


# information-theoretic priority per constraint prototype: rho = |fixed| / |space|
# (lower rho -> larger info gain -log(rho) -> higher priority). Values follow the
# paper's ordering of common MIPLIB constraint types (heuristic constants).
TYPE_RHO = {
    "set_partitioning": 0.15,   # sum x == 1  (equality, tightest)
    "assignment": 0.20,         # equality over 0/1
    "set_covering": 0.35,       # sum x >= 1
    "set_packing": 0.40,        # sum x <= 1
    "cardinality": 0.55,        # sum x (<=|>=) k, unit coeffs, k>1
    "knapsack": 0.70,           # general positive coeffs, <=
    "invariant_knapsack": 0.75,
    "mixed": 0.85,
    "general": 0.95,            # everything else (lowest priority)
}


def get_sorted_constraints(ins_path):
    """Return the linear constraints in get_a_new2 order: list of
    (name, coeff_dict, rhs, sense)."""
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_path)
    cons = [c for c in m.getConss() if len(m.getValsLinear(c)) > 0]
    cons_map = sorted([[c, len(m.getValsLinear(c))] for c in cons],
                      key=lambda x: [x[1], str(x[0])])
    cons = [x[0] for x in cons_map]
    out = []
    for c in cons:
        coeff = dict(m.getValsLinear(c))
        rhs, lhs = m.getRhs(c), m.getLhs(c)
        if rhs == lhs:
            sense, b = 2, rhs
        elif rhs >= 1e20:
            sense, b = 1, lhs
        else:
            sense, b = 0, rhs
        out.append((str(c), coeff, float(b), sense))
    return out


def classify_constraint(coeff_dict, rhs, sense):
    """Classify one constraint into a MIPLIB-style prototype (heuristic)."""
    vals = np.array(list(coeff_dict.values()), dtype=np.float64)
    if vals.size == 0:
        return "general"
    unit = np.allclose(np.abs(vals), 1.0)
    positive = np.all(vals > 0)
    if sense == 2:  # equality
        if unit and abs(rhs - 1.0) < 1e-6:
            return "set_partitioning"
        return "assignment"
    if unit:
        if sense == 1 and abs(rhs - 1.0) < 1e-6:
            return "set_covering"
        if sense == 0 and abs(rhs - 1.0) < 1e-6:
            return "set_packing"
        return "cardinality"
    if sense == 0 and positive:
        return "knapsack"
    if positive:
        return "invariant_knapsack"
    return "mixed"


def type_priority(t):
    """Info gain -log(rho); higher = more critical."""
    rho = TYPE_RHO.get(t, TYPE_RHO["general"])
    return float(-np.log(rho))


def compute_ctc_labels(ins_path, x_by_name, tol=1e-6, use_tcp=True,
                       n_min=1, frac_lo=0.05, frac_hi=0.95):
    """Per-constraint critical-tight-constraint labels aligned to get_a_new2's
    constraint order (length = #real constraints; the objective node row, added
    as the last constraint node by get_a_new2, is not included here -- callers
    append a 0 for it).

    A constraint is TIGHT at x if |a^T x - b| <= tol. If use_tcp, only tight
    constraints whose prototype is 'selected' by the JUDGE rule (enough tight
    members and a healthy tight fraction) are labelled critical, and selection
    proceeds by descending type priority.

    Returns (labels[ncons] float32, types[ncons] list, tight[ncons] float32).
    """
    cons = get_sorted_constraints(ins_path)
    ncons = len(cons)
    tight = np.zeros(ncons, dtype=np.float32)
    types = []
    for i, (_name, coeff, b, sense) in enumerate(cons):
        act = sum(v * x_by_name.get(k, 0.0) for k, v in coeff.items())
        denom = max(1.0, abs(b))
        # Only INEQUALITIES are reducible (an equality is already tight and cannot
        # be tightened further), and the evaluator only tightens sense 0/1, so we
        # label critical-tight constraints among inequalities only -- keeping the
        # training target consistent with what is actually used at solve time.
        if sense in (0, 1) and abs(act - b) <= tol * denom:
            tight[i] = 1.0
        types.append(classify_constraint(coeff, b, sense))

    if not use_tcp:
        return tight.copy(), types, tight

    # TCP JUDGE: rank prototypes by priority, keep those with enough tight members
    # and a moderate tight fraction; CTC = tight AND type-selected.
    labels = np.zeros(ncons, dtype=np.float32)
    by_type = {}
    for i, t in enumerate(types):
        by_type.setdefault(t, []).append(i)
    for t in sorted(by_type, key=type_priority, reverse=True):
        idx = by_type[t]
        tcount = int(tight[idx].sum())
        frac = tcount / max(1, len(idx))
        if tcount >= n_min and frac_lo <= frac <= frac_hi:
            for i in idx:
                if tight[i] > 0.5:
                    labels[i] = 1.0
    # fallback: if JUDGE selected nothing, use raw tightness
    if labels.sum() == 0:
        labels = tight.copy()
    return labels, types, tight

"""
CL-LNS data collection with a Local-Branching (LB) expert.

For each training instance we run a few LNS steps guided by an LB expert and
record, per state, the destroy ACTIONS to imitate contrastively:

  * positive action  = the subset of binary variables the LB optimum flips
    relative to the current incumbent (the expert's destroy neighborhood);
  * negative actions  = random perturbations of that subset (destroy sets that
    are not the expert's choice).

Each state also stores a W-step incumbent-value history per variable (graph
order), used as extra variable features so the policy is incumbent-conditioned.

This is the dependency-light variant (no Khalil root features / Ecole): the LB
sub-ILP is solved with the existing solver wrapper. Kept intentionally small so
it runs on modest hardware; scale probe/step counts up for real training.
"""

import os
import pickle

import numpy as np

from src.solver.solver_utils import SOLVER_CLASSES
from src.utils.utils import get_a_new2


def _solve_values(solver, time_limit, threads, solver_name):
    solver.solve(means=solver_name, log_file='', time_limit=time_limit, threads=threads)
    vals = {}
    for v in solver.get_vars():
        try:
            vals[solver.varname(v)] = float(solver.varval(v))
        except Exception:
            pass
    return vals


def _lb_step(ins_path, incumbent, bin_names, k, time_limit, threads, solver_name):
    """Local-Branching sub-ILP: original problem + Hamming ball of radius k around
    the incumbent, over the BINARY variables only. Returns the new (improved)
    assignment as a name->value dict."""
    s = SOLVER_CLASSES[solver_name]()
    s.hide_output_to_console()
    s.load_model(ins_path)
    bmap = {s.varname(v): v for v in s.get_vars()}
    flip_terms = []
    for nm in bin_names:
        v = bmap.get(nm)
        if v is None:
            continue
        val = incumbent.get(nm, 0.0)
        flip_terms.append(v if round(val) == 0 else (1 - v))
    if flip_terms:
        s.add_constraint(sum(flip_terms) <= k, name="hamming_ball")
    return _solve_values(s, time_limit, threads, solver_name)


def collect_instance(ins_path, n_states=3, k_frac=0.2, window=3,
                     init_time=10, lb_time=10, n_neg=8, threads=1,
                     solver_name='gurobi', seed=0):
    rng = np.random.default_rng(seed)
    bg = get_a_new2(ins_path)
    A, v_map, v_nodes, c_nodes, b_vars = bg
    names_graph = list(v_map)                       # graph order
    b_idx = b_vars.tolist()                         # binary var graph indices
    nb = len(b_idx)
    if nb == 0:
        return []

    solver = SOLVER_CLASSES[solver_name]()
    solver.hide_output_to_console()
    solver.load_model(ins_path)
    incumbent = _solve_values(solver, init_time, threads, solver_name)
    if not incumbent:
        return []
    bin_names = {names_graph[g] for g in b_idx}

    k = max(1, int(round(k_frac * nb)))
    hist = [np.array([round(incumbent.get(names_graph[g], 0.0)) for g in range(len(names_graph))],
                     dtype=np.float32)]
    states = []
    for _step in range(n_states):
        x_lb = _lb_step(ins_path, incumbent, bin_names, k, lb_time, threads, solver_name)
        if not x_lb:
            break
        # positive destroy action = binary vars LB flipped (graph order -> binary order)
        flip_graph = np.array(
            [1.0 if round(x_lb.get(names_graph[g], 0.0)) != round(incumbent.get(names_graph[g], 0.0))
             else 0.0 for g in range(len(names_graph))], dtype=np.float32)
        pos = flip_graph[b_idx]                     # [nb]
        if pos.sum() == 0:                          # LB found no improving flip; stop
            break

        # negatives: CARDINALITY-PRESERVING perturbations of the expert subset
        # (swap r destroyed vars for r non-destroyed) so the contrastive signal is
        # about WHICH variables to destroy, not how many.
        ones = np.where(pos > 0.5)[0]
        zeros = np.where(pos < 0.5)[0]
        negs = []
        for _ in range(n_neg):
            a = pos.copy()
            r = max(1, int(round(0.3 * max(len(ones), 1))))
            if len(ones) > 0:
                a[rng.choice(ones, size=min(r, len(ones)), replace=False)] = 0.0
            if len(zeros) > 0:
                a[rng.choice(zeros, size=min(r, len(zeros)), replace=False)] = 1.0
            if a.sum() > 0 and not np.array_equal(a, pos):
                negs.append(a)
        if not negs:
            continue

        # incumbent-value window (graph order), most recent W
        w = hist[-window:]
        while len(w) < window:
            w = [w[0]] + w
        win = np.stack(w[-window:], axis=1)         # [n_var, window]

        states.append({
            'bg': bg,
            'window': win.astype(np.float32),
            'pos_actions': pos[None, :].astype(np.float32),   # [1, nb]
            'neg_actions': np.stack(negs, axis=0).astype(np.float32),  # [N, nb]
        })

        incumbent = {nm: round(x_lb.get(nm, incumbent.get(nm, 0.0))) for nm in incumbent}
        hist.append(np.array([incumbent.get(names_graph[g], 0.0) for g in range(len(names_graph))],
                             dtype=np.float32))
    return states


def collect_dataset(input_dir, output_dir, solver_name='gurobi', **kw):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(('.lp', '.mps', '.lp.gz', '.mps.gz'))]
    total = 0
    for fn in files:
        path = os.path.join(input_dir, fn)
        try:
            states = collect_instance(path, solver_name=solver_name, **kw)
        except Exception as e:
            print(f"[CL-LNS] error on {fn}: {e}")
            continue
        base = os.path.splitext(fn)[0]
        for si, st in enumerate(states):
            with open(os.path.join(output_dir, f"{base}_s{si}.pkl"), "wb") as f:
                pickle.dump(st, f)
        total += len(states)
        print(f"[CL-LNS] {fn}: {len(states)} states")
    print(f"[CL-LNS] collected {total} states -> {output_dir}")
    return total

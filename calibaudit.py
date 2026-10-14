#!/usr/bin/env python3
"""calibaudit: is a binary probability forecast honest, and does a recalibration survive days it was not fitted on?

    python3 calibaudit.py decisions.csv [--budget 0.05] [--gate p|z|calibrated] [--link logit|probit] [--folds 5] [--json out.json]

Input is a CSV with the header p,outcome,day: p is the forecast probability that outcome is 1, outcome is 0 or 1, and
day labels the cluster rows belong to. Rows of one day are held out and resampled together, so a correlated day counts
once, not once per row. ISO dates sort in time order, which the forward check relies on.

A port of the evaluation layer of jaredpalmer/kev (kev/metrics.py, kev/calibrate.py) and of the abstention cut in
NandhaKishorM/laya (laya/calibrate.py), both Apache-2.0 (see NOTICE), rewritten for binary forecasts. Standard library only.
"""
import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from decimal import ROUND_FLOOR, Decimal
from statistics import NormalDist

EPS = 1e-9   # kev EPSILON: p is clipped to [EPS, 1 - EPS] before a logit or a log
T_GRID = [math.exp(math.log(0.25) + i * math.log(16) / 120) for i in range(121)]   # kev TEMPERATURE_FIT: 121-point log grid 0.25..4
GATES = ("p", "z", "calibrated")
LINKS = ("logit", "probit")
ND = NormalDist()


def clip(p):
    return min(1 - EPS, max(EPS, p))


def logit(p):
    p = clip(p)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def softplus(x):
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def probit(p):
    return ND.inv_cdf(clip(p))


def zval(p):
    """z = Phi^-1(p) as the z gate and the reliability table show it, clipped at 1e-15 on both sides."""
    return ND.inv_cdf(min(max(p, 1e-15), 1 - 1e-15))


def log_phi(u):
    return math.log(0.5 * math.erfc(-u / math.sqrt(2)))


def temper(p, t, link="logit"):
    """p with its logit (or probit) divided by t: t > 1 pulls an over-confident forecast toward 0.5, t < 1 pushes it out.
    On the probit link with p = Phi(z), tempering is Phi(z / t): the spread behind z was t times what the forecast assumed."""
    return sigmoid(logit(p) / t) if link == "logit" else ND.cdf(probit(p) / t)


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = {"p", "outcome", "day"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing column(s): {', '.join(sorted(missing))} (header must be p,outcome,day)")
        for line, rec in enumerate(reader, start=2):
            try:
                p, y = float(rec["p"]), float(rec["outcome"])
            except (TypeError, ValueError):
                raise ValueError(f"line {line}: p and outcome must be numbers") from None
            if not 0 <= p <= 1:
                raise ValueError(f"line {line}: p={rec['p']} is outside [0, 1]")
            if y not in (0.0, 1.0):
                raise ValueError(f"line {line}: outcome={rec['outcome']} is not 0 or 1")
            day = (rec["day"] or "").strip()
            if not day:
                raise ValueError(f"line {line}: empty day")
            rows.append((p, int(y), day))
    if not rows:
        raise ValueError("no rows")
    return rows


def brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)


def logloss(ps, ys):
    return -sum(math.log(clip(p)) if y else math.log(1 - clip(p)) for p, y in zip(ps, ys)) / len(ps)


def bins_of(ps, k, mode):
    """Index lists of k bins: equal-mass (sorted order cut into k equal counts) or equal-width on [0, 1] (kev's ece).
    An equal-mass edge never splits rows with the same p: it moves to the end of the tie, so a bin can be larger and
    there can be fewer than k bins (forecasts in whole cents have few distinct values)."""
    if mode == "width":
        out = [[] for _ in range(k)]
        for i, p in enumerate(ps):
            out[min(int(p * k), k - 1)].append(i)
        return [b for b in out if b]
    order = sorted(range(len(ps)), key=ps.__getitem__)
    out, start = [], 0
    for j in range(1, k + 1):
        end = max(j * len(ps) // k, start)
        while 0 < end < len(ps) and ps[order[end]] == ps[order[end - 1]]:
            end += 1
        if end > start:
            out.append(order[start:end])
            start = end
    return out


def reliability(ps, ys, k=10, mode="mass"):
    table = []
    for b in bins_of(ps, k, mode):
        table.append({"n": len(b), "lo": min(ps[i] for i in b), "hi": max(ps[i] for i in b),
                      "mean_p": sum(ps[i] for i in b) / len(b), "rate": sum(ys[i] for i in b) / len(b)})
    return table


def ece(ps, ys, k=10, mode="mass"):
    return sum(r["n"] * abs(r["rate"] - r["mean_p"]) for r in reliability(ps, ys, k, mode)) / len(ps)


def scores(ps, ys, k, mode):
    return {"brier": brier(ps, ys), "logloss": logloss(ps, ys), "ece": ece(ps, ys, k, mode),
            "mean_p": sum(ps) / len(ps), "rate": sum(ys) / len(ys)}


LOSS_LO, LOSS_HI = -math.log(1 - EPS), -math.log(EPS)


def fit_temperature(ps, ys, link="logit"):
    """kev fit_temperature for the binary case: the grid point with the lowest mean log-loss. The loss is the one
    logloss() reports, the tempered p clipped to [EPS, 1 - EPS], so the chosen T is the grid minimum of the printed
    number. Identical (logit, outcome) pairs are counted once with a weight, which changes nothing but the run time."""
    to = logit if link == "logit" else probit
    pairs = Counter((to(p), y) for p, y in zip(ps, ys))
    total = sum(pairs.values())
    if link == "logit":
        one = lambda x, y: softplus(-x if y else x)
    else:
        one = lambda x, y: -log_phi(x if y else -x)
    loss = lambda t: sum(c * min(max(one(v / t, y), LOSS_LO), LOSS_HI) for (v, y), c in pairs.items())
    losses = [loss(t) / total for t in T_GRID]
    return T_GRID[losses.index(min(losses))]


def day_folds(days, k, seed):
    """Fold id per day: days shuffled with the seed, then dealt round-robin, so no day sits in two folds."""
    uniq = sorted(set(days))
    if len(uniq) < k:
        raise ValueError(f"{len(uniq)} distinct days, fewer than {k} folds")
    rng = random.Random(seed)
    rng.shuffle(uniq)
    return {d: i % k for i, d in enumerate(uniq)}


def out_of_fold(ps, ys, days, k, seed, link="logit"):
    """Each fold's p tempered by a temperature fitted on the other folds' days (kev out_of_fold_rows)."""
    fold = day_folds(days, k, seed)
    cal, temps = [None] * len(ps), []
    for f in range(k):
        train = [i for i, d in enumerate(days) if fold[d] != f]
        t = fit_temperature([ps[i] for i in train], [ys[i] for i in train], link)
        temps.append(t)
        for i, d in enumerate(days):
            if fold[d] == f:
                cal[i] = temper(ps[i], t, link)
    return cal, temps, fold


def quantile(values, q):
    v = sorted(values)
    pos = q * (len(v) - 1)
    lo = int(pos)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (pos - lo)


def day_bootstrap(days, num, den, samples, seed):
    """sum(num) / sum(den) with a 95% percentile interval from resampling whole days with replacement. Brier and
    log-loss deltas and error rates are ratios of per-row sums, so each resample costs one pass over the days."""
    agg = defaultdict(lambda: [0.0, 0.0])
    for d, a, b in zip(days, num, den):
        agg[d][0] += a
        agg[d][1] += b
    keys = sorted(agg)
    total_den = sum(agg[d][1] for d in keys)
    point = sum(agg[d][0] for d in keys) / total_den if total_den else None
    rng, draws = random.Random(seed), []
    for _ in range(samples):
        a = b = 0.0
        for _ in keys:
            x = agg[keys[rng.randrange(len(keys))]]
            a += x[0]
            b += x[1]
        if b:
            draws.append(a / b)
    if point is None or not draws:
        return {"value": point, "ci95": None}
    return {"value": point, "ci95": [quantile(draws, 0.025), quantile(draws, 0.975)], "resamples_used": len(draws)}


def paired_delta(days, ys, cand, ref, samples, seed):
    """Brier and log-loss of cand minus ref, paired by row, CI over days. Negative = cand is better."""
    ones = [1.0] * len(ys)
    d_brier = [(c - y) ** 2 - (r - y) ** 2 for c, r, y in zip(cand, ref, ys)]
    d_ll = [(-math.log(clip(c)) + math.log(clip(r))) if y else (-math.log(1 - clip(c)) + math.log(1 - clip(r)))
            for c, r, y in zip(cand, ref, ys)]
    out = {"brier": day_bootstrap(days, d_brier, ones, samples, seed), "logloss": day_bootstrap(days, d_ll, ones, samples, seed)}
    for v in out.values():
        v["separated"] = bool(v["ci95"] and (v["ci95"][1] < 0 or v["ci95"][0] > 0))
    return out


def gate_values(ps, gate, t=1.0, link="logit"):
    """Confidence on the scale the live gate compares: p itself, z = Phi^-1(p) when p came from Phi(z), or p after the
    temperature. A binary forecast predicts outcome 1 when p >= 0.5, so its confidence is max(p, 1 - p)."""
    conf = [max(p, 1 - p) for p in ps]
    if gate == "calibrated":
        return [temper(c, t, link) for c in conf]
    if gate == "z":
        return [zval(c) for c in conf]
    return conf


def select_cut(values, correct, budget, conservative=True):
    """laya _select_abstention_threshold: the lowest cut whose accepted set {v >= cut} keeps the error <= budget, with
    every row tied at a level counted before the level is judged. conservative adds one pseudo-error, so a short lucky
    run at the top does not clear the gate. None when no cut holds the budget."""
    order = sorted(zip(values, correct), key=lambda x: -x[0])
    best, n_acc, n_err, i = None, 0, 0, 0
    for level in sorted(set(values), reverse=True):
        while i < len(order) and order[i][0] >= level:
            n_acc += 1
            n_err += 0 if order[i][1] else 1
            i += 1
        rate = (n_err + 1) / (n_acc + 1) if conservative else n_err / n_acc
        if rate <= budget:
            best = level
    return best


def cut_text(cut, values, places):
    """The cut rounded down to `places` decimals, with more decimals only when the rounded number would also admit
    the next value below the cut. The printed number then accepts exactly the rows the exact cut accepts: rounding
    to nearest can admit rows just below the cut or drop the ones sitting on it."""
    if cut is None:
        return None
    below = max((v for v in values if v < cut), default=-math.inf)
    exact = Decimal(repr(cut))
    for n in range(places, 18):
        x = exact.quantize(Decimal(1).scaleb(-n), rounding=ROUND_FLOOR)
        if below < float(x) <= cut:
            return str(x)
    return repr(cut)


def cut_report(ps, ys, days, budget, gate, link, t_all, fold, temps, samples, seed, forward_days):
    correct = [(p >= 0.5) == bool(y) for p, y in zip(ps, ys)]
    wrong = [0.0 if c else 1.0 for c in correct]

    def held_out(train_idx, test_idx, t):
        train_vals = gate_values([ps[i] for i in train_idx], gate, t, link)
        cut = select_cut(train_vals, [correct[i] for i in train_idx], budget)
        text = cut_text(cut, train_vals, places)
        vals = gate_values([ps[i] for i in test_idx], gate, t, link)
        acc = [1.0 if text is not None and v >= float(text) else 0.0 for v in vals]   # judge the printed number
        return cut, text, acc

    def judged(acc, idx, cut=None, text=None):
        errs = [acc[j] * wrong[i] for j, i in enumerate(idx)]
        return {"cut": cut, "cut_text": text, "accepted": int(sum(acc)), "errors": int(sum(errs)), "coverage": sum(acc) / len(acc),
                "accepted_days": len({days[i] for j, i in enumerate(idx) if acc[j]}),
                "error": day_bootstrap([days[i] for i in idx], errs, acc, samples, seed)}

    places = 3 if gate == "z" else 5
    values = gate_values(ps, gate, t_all, link)
    cut = select_cut(values, correct, budget)
    every = range(len(ps))
    two_sided = any(p < 0.5 for p in ps)
    label = {"p": "max(p, 1-p)" if two_sided else "p", "z": "|z|" if two_sided else "z",
             "calibrated": "calibrated max(p, 1-p)" if two_sided else "calibrated p"}[gate]
    out = {"budget": budget, "gate": gate, "label": label,
           **judged([1.0 if cut is not None and v >= cut else 0.0 for v in values], every, cut, cut_text(cut, values, places))}

    # out-of-fold: each fold's cut is fitted on the other folds' days (and their temperature) and judged on its own
    k = len(temps)
    acc_oof, cuts, texts = [0.0] * len(ps), [], []
    for f in range(k):
        train = [i for i, d in enumerate(days) if fold[d] != f]
        test = [i for i, d in enumerate(days) if fold[d] == f]
        c, text, a = held_out(train, test, temps[f])
        cuts.append(c)
        texts.append(text)
        for i, v in zip(test, a):
            acc_oof[i] = v
    out["oof"] = {"cuts": cuts, "cut_texts": texts, **judged(acc_oof, every)}

    if forward_days:
        first, last = forward_days
        train = [i for i, d in enumerate(days) if d in first]
        test = [i for i, d in enumerate(days) if d in last]
        t_fwd = fit_temperature([ps[i] for i in train], [ys[i] for i in train], link)
        c, text, a = held_out(train, test, t_fwd)
        out["forward"] = judged(a, test, c, text)
    return out


def audit(rows, budget=0.05, gate="p", folds=5, samples=2000, seed=0, bins=10, bin_mode="mass", link="logit"):
    if gate not in GATES or link not in LINKS:
        raise ValueError(f"gate must be one of {', '.join(GATES)} and link one of {', '.join(LINKS)}")
    ps, ys, days = [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
    uniq = sorted(set(days))
    t_all = fit_temperature(ps, ys, link)
    cal, temps, fold = out_of_fold(ps, ys, days, folds, seed, link)
    report = {"n": len(rows), "days": len(uniq), "first_day": uniq[0], "last_day": uniq[-1],
              "raw": scores(ps, ys, bins, bin_mode), "oof": scores(cal, ys, bins, bin_mode),
              "mean_p_minus_rate": day_bootstrap(days, [p - y for p, y in zip(ps, ys)], [1.0] * len(ps), samples, seed),
              "temperature": t_all, "fold_temperatures": temps,
              "oof_minus_raw": paired_delta(days, ys, cal, ps, samples, seed),
              "reliability": reliability(ps, ys, bins, bin_mode), "bins": bins, "bin_mode": bin_mode,
              "link": link, "folds": folds, "samples": samples, "seed": seed}
    forward_days = None
    if len(uniq) >= 6:
        split = len(uniq) * 2 // 3
        first, last = set(uniq[:split]), set(uniq[split:])
        forward_days = (first, last)
        train = [i for i, d in enumerate(days) if d in first]
        test = [i for i, d in enumerate(days) if d in last]
        t_fwd = fit_temperature([ps[i] for i in train], [ys[i] for i in train], link)
        report["forward"] = {"fit_days": [uniq[0], uniq[split - 1]], "test_days": [uniq[split], uniq[-1]], "temperature": t_fwd,
                             "test_minus_raw": paired_delta([days[i] for i in test], [ys[i] for i in test],
                                                            [temper(ps[i], t_fwd, link) for i in test], [ps[i] for i in test], samples, seed)}
    report["cut"] = cut_report(ps, ys, days, budget, gate, link, t_all, fold, temps, samples, seed, forward_days)
    report["warnings"] = warnings(report, ys, days)
    return report


def warnings(rep, ys, days):
    w = []
    if rep["days"] < 20:
        w.append(f"only {rep['days']} days: intervals are wide and every fold holds few days")
    minority = min(sum(ys), len(ys) - sum(ys))
    if minority < 30:
        w.append(f"only {minority} rows of the rarer outcome: calibration at the confident end rests on a handful of events")
    edge = [t for t in [rep["temperature"], *rep["fold_temperatures"]] if t in (T_GRID[0], T_GRID[-1])]
    if edge:
        w.append("a fitted temperature sits at the grid edge (0.25 or 4): the best value may lie beyond it")
    big = Counter(days).most_common(1)[0]
    if big[1] > 0.25 * len(days):
        w.append(f"day {big[0]} holds {big[1] / len(days):.0%} of the rows")
    return w


def _ci(b, fmt="{:+.4f}"):
    if b["value"] is None:
        return "n/a"
    s = fmt.format(b["value"])
    return s + (" [" + ", ".join(fmt.format(x) for x in b["ci95"]) + "]" if b["ci95"] else "")


def _err(r):
    """Error among accepted rows with its day interval. With no errors every resample is 0 and the interval collapses.
    The rule of three then gives two different 95% bounds: on the share of ROWS with an error if rows are independent,
    and on the share of DAYS with any error if days are independent. With correlated rows only the second holds, and
    it bounds days, not rows: one big day can carry most of the row risk."""
    if r["accepted"] == 0:
        return "nothing accepted"
    if r["errors"] == 0:
        return (f"0 of {r['accepted']} on {r['accepted_days']} days (no interval; rule of three: under "
                f"{3 / r['accepted']:.2%} of rows if rows are independent, under {min(1, 3 / r['accepted_days']):.2%} "
                f"of days if days are)")
    return f"{r['errors']} of {r['accepted']} = {_ci(r['error'], '{:.2%}')}"


def _cut(r, label):
    return "none holds the budget" if r["cut_text"] is None else f"{label} >= {r['cut_text']}"


def format_report(rep):
    raw, oof = rep["raw"], rep["oof"]
    lines = [f"rows {rep['n']}, days {rep['days']} ({rep['first_day']} .. {rep['last_day']}), outcome rate {raw['rate']:.4f}, "
             f"mean p {raw['mean_p']:.4f}",
             f"mean p minus outcome rate {_ci(rep['mean_p_minus_rate'])} (95% CI over days; > 0: p promises more than happens)", "",
             f"{'':28}{'Brier':>9}{'log-loss':>10}{'ECE':>9}",
             f"{'raw p':28}{raw['brier']:9.5f}{raw['logloss']:10.5f}{raw['ece']:9.5f}",
             f"{'out-of-fold tempered p':28}{oof['brier']:9.5f}{oof['logloss']:10.5f}{oof['ece']:9.5f}", "",
             f"{rep['link']} temperature fitted on all days: T = {rep['temperature']:.3f}   (T > 1: p is over-confident, T < 1: under-confident)",
             f"per fold ({rep['folds']} day-disjoint folds): " + ", ".join(f"{t:.3f}" for t in rep["fold_temperatures"])]
    d = rep["oof_minus_raw"]
    lines.append(f"out-of-fold minus raw, 95% CI over days ({rep['samples']} resamples): Brier {_ci(d['brier'], '{:+.5f}')}"
                 f"{'  separated' if d['brier']['separated'] else '  not separated'}, log-loss {_ci(d['logloss'])}")
    if "forward" in rep:
        f = rep["forward"]
        fd = f["test_minus_raw"]
        lines.append(f"forward: T = {f['temperature']:.3f} fitted on {f['fit_days'][0]} .. {f['fit_days'][1]}, tested on "
                     f"{f['test_days'][0]} .. {f['test_days'][1]}: Brier {_ci(fd['brier'], '{:+.5f}')}"
                     f"{'  separated' if fd['brier']['separated'] else '  not separated'}")
    scale = "z" if rep["cut"]["gate"] == "z" else "p"
    lines += ["", f"reliability, raw p, {rep['bins']} {'equal-mass' if rep['bin_mode'] == 'mass' else 'equal-width'} bins",
              f"{'n':>7}  {scale + ' range':>19}{'mean p':>9}{'hit rate':>10}{'gap':>9}"]
    for r in rep["reliability"]:
        lo, hi = (r["lo"], r["hi"]) if scale == "p" else (zval(r["lo"]), zval(r["hi"]))
        rng = f"{lo:.4f}..{hi:.4f}" if scale == "p" else f"{lo:.3f}..{hi:.3f}"
        lines.append(f"{r['n']:7d}  {rng:>19}{r['mean_p']:9.4f}{r['rate']:10.4f}{r['rate'] - r['mean_p']:+9.4f}")
    c = rep["cut"]
    lines += ["", f"error budget {c['budget']:.1%}, cut on the gate's own scale ({c['label']}), conservative +1 rule",
              f"  fitted on all days: {_cut(c, c['label'])}: accepts {c['coverage']:.1%} of rows, errors {_err(c)}",
              f"  out-of-fold (cut fitted on the other days): accepts {c['oof']['coverage']:.1%}, errors {_err(c['oof'])}"
              f"; fold cuts " + ", ".join("none" if x is None else x for x in c["oof"]["cut_texts"])]
    if "forward" in c:
        lines.append(f"  forward (fitted on the first days, judged on the last): {_cut(c['forward'], c['label'])}: "
                     f"accepts {c['forward']['coverage']:.1%}, errors {_err(c['forward'])}")
    if rep["warnings"]:
        lines += [""] + [f"warning: {x}" for x in rep["warnings"]]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", help="CSV with header p,outcome,day")
    ap.add_argument("--budget", type=float, default=0.05, help="tolerated error among accepted rows (default 0.05)")
    ap.add_argument("--gate", choices=GATES, default="p",
                    help="scale the live gate compares: p, z (p came from Phi(z)), or calibrated (p after the temperature)")
    ap.add_argument("--link", choices=LINKS, default="logit",
                    help="temperature on the logit (kev) or on the probit: with p = Phi(z), probit T is the factor the spread was underestimated by")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--samples", type=int, default=2000, help="bootstrap resamples of days")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--equal-width", action="store_true", help="equal-width bins on [0, 1] instead of equal-mass")
    ap.add_argument("--json", help="also write the full report here")
    a = ap.parse_args(argv)
    if not 0 <= a.budget <= 1 or a.folds < 2 or a.samples < 1 or a.bins < 1:
        ap.error("need 0 <= budget <= 1, folds >= 2, samples >= 1, bins >= 1")
    try:
        rep = audit(read_rows(a.csv), a.budget, a.gate, a.folds, a.samples, a.seed, a.bins, "width" if a.equal_width else "mass", a.link)
    except (OSError, ValueError) as e:
        print(f"calibaudit: {e}", file=sys.stderr)
        return 2
    print(format_report(rep))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

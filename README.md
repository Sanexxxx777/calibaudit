# calibaudit

**Check whether a probability forecast means what it says, and whether a fix fitted on past days holds on the next ones.**

I had a model that turns a distance in standard deviations into a probability, p = Φ(z), and a gate that
acts when z clears a threshold. Two questions mattered. Does p = 0.97 really win 97 times out of 100?
And would any correction I fitted on last month still hold next week, or is it one more short-window
optimum? Averages over thousands of rows don't answer either question when decisions on the same day move
together. `calibaudit` scores the forecast, fits one temperature on days it then holds out, and reports
every improvement with an interval computed over whole days.

```bash
python3 calibaudit.py decisions.csv
python3 calibaudit.py decisions.csv --gate z --link probit --budget 0.05 --json report.json
```

The input is a CSV with the header `p,outcome,day`. `p` is the forecast probability that `outcome` is 1,
`outcome` is 0 or 1, and `day` labels the cluster. Rows of one day are held out and resampled together.
Python 3.9+, standard library only. One file, about 430 lines. 6,000 rows take under a second.

## How it works

- **Scores.** Brier, log-loss, ECE, and a 10-bin reliability table. The bins hold equal counts by default,
  because a forecast that only acts above 0.9 puts every row into one or two equal-width bins
  (`--equal-width` restores kev's bins). A bin edge never splits rows with the same p. Forecasts in
  whole cents have few distinct values, and a split tie inflated ECE from 0.10 to 0.26 on a 10-row test
  case before this rule was added.
- **Mean gap.** Mean p minus the outcome rate, with a 95% interval over days. This is the plainest number
  the tool gives: how much the forecast promises beyond what happens.
- **Temperature.** One temperature, chosen as kev chooses it: the lowest mean log-loss over 121
  log-spaced points between 0.25 and 4. `--link logit` (default) divides the logit. `--link probit`
  divides Φ⁻¹(p) instead, so when p = Φ(z), T is the factor by which the spread behind z was too small.
- **Held-out days.** The days are dealt into 5 folds. Each fold gets the temperature fitted on the other
  four, so no prediction is scored by a fit that saw its own day. The change in Brier and log-loss
  against the raw forecast comes with a paired 95% interval from resampling whole days, 2,000 times.
  `separated` means the interval excludes zero.
- **Forward split.** A temperature fitted on the first two thirds of the days (in sorted order, so ISO
  dates work) is judged on the last third, with the same interval.
- **Error-budget cut.** This is the lowest confidence that keeps the error among accepted rows within
  `--budget`. It follows laya's conservative rule: one pseudo-error is added, so a short clean run at the
  top cannot clear the gate, and a tied group is judged as a whole. The cut is fitted on the scale the
  gate compares (`--gate p`, `z` or `calibrated`). It is then checked twice on held-out data: out of
  fold, and forward. When no errors are observed, the day interval collapses to [0, 0], so the report
  prints a rule-of-three bound (3/n) instead.

## What it does not do

- **It cannot tell a spread that is too small everywhere from rare jumps.** Both look like
  over-confidence: the `probit15` example below fits T = 1.48, the `jumps` example T = 1.18. If you log
  the raw move behind each forecast, standardise it and look at its spread and tails directly. Binary
  outcomes carry too little information for that.
- **It fits one temperature, nothing more flexible.** There is no bias term (Platt), no isotonic
  regression, no histogram binning. If the hit rate stays flat while p climbs, a temperature cannot
  follow it. The out-of-fold change then comes out not separated, or separated in the wrong direction
  (worse). That outcome is a finding about the model, not a failed run.
- **It does not extrapolate.** If a gate already filtered the rows (say z ≥ 1.5), every number is
  conditional on that filter and says nothing about a lower threshold.
- **It does not know how the rows were selected.** If the rows are executed decisions, such as filled
  orders, the calibration holds only for decisions that got executed.
- **It weighs every row once.** There is no stake or size column.
- **It does not price anything.** The cut limits the error rate, not the loss. A decision bought at 0.93
  breaks even at 7% error, so a 5% budget is a convention, not a profit line.
- **It assumes days are close to independent.** With 15 to 20 days the intervals are wide, and the
  report says so. ECE has no interval because it is not a per-day sum. Regime changes are tested only by
  the one forward split.

## Where it came from

The method is a port of the evaluation layer of
[`jaredpalmer/kev`](https://github.com/jaredpalmer/kev) (Apache-2.0), `kev/metrics.py` and
`kev/calibrate.py`: the temperature grid, the out-of-fold fit on disjoint groups, the clustered paired
bootstrap, and coverage at an error budget. kev applies these to multi-class model logits. This tool
rewrites them for a binary forecast in a plain CSV, with days as the clusters.

The cut comes from [`NandhaKishorM/laya`](https://github.com/NandhaKishorM/laya) (Apache-2.0),
`laya/calibrate.py`. Its docstring records why the scale matters. A cut fitted before histogram binning
held 9.8% error on the scale it was fitted on. Compared against binned confidences, the same number
admitted 94.5% of answers at 25.6% error. `tests/test_calibaudit.py` covers the same failure: a cut
chosen on the calibrated scale and compared against raw p admits more rows than it should. Both credits
and the list of changes are in `NOTICE`.

## Measured numbers

The figures come from the three files `examples/synthetic.py` writes: 40 days × 150 decisions each,
p between 0.6 and 0.999, all seeded.

| File | Truth | Fitted T | Mean p − rate | Brier change out of fold | ECE raw → out of fold |
|---|---|---|---|---|---|
| `honest.csv` | p | 1.02 (logit) | −0.001 [−0.011, +0.009] | +0.00003 [−0.00005, +0.00010], not separated | 0.016 → 0.014 |
| `probit15.csv` | Φ(z / 1.5) | 1.48 (probit) | +0.066 [+0.057, +0.076] | −0.0049 [−0.0063, −0.0035], separated | 0.066 → 0.018 |
| `jumps.csv` | 0.9 Φ(z) + 0.1 Φ(z / 4) | 1.18 (probit) | +0.026 [+0.016, +0.036] | −0.0007 [−0.0013, −0.0001], separated | 0.026 → 0.009 |

The honest forecast comes back with T near 1 and no improvement. The forecast whose spread is 1.5 times
too small comes back at 1.48, one grid step from the truth (grid points sit about 2.3% apart). On the
jumps file the tool correctly reports over-confidence, but T alone does not tell it apart from a spread
that is too small everywhere.

## Reproduce

```bash
python3 examples/synthetic.py /tmp/calib_demo
python3 calibaudit.py /tmp/calib_demo/honest.csv
python3 calibaudit.py /tmp/calib_demo/probit15.csv --gate z --link probit
python3 calibaudit.py /tmp/calib_demo/jumps.csv --gate z --link probit
```

## Tests

```bash
python3 tests/test_calibaudit.py
```

The suite covers hand-computed Brier, log-loss and ECE, and ties held inside one bin. It checks that a
temperature is recovered for over- and under-confident synthetic forecasts on both links, and that the
out-of-fold prediction for a day does not change when that day's outcomes are flipped. Day clusters must
widen the interval when days are correlated. The cut tests cover the conservative rule, ties, the
rule-of-three line, and the scale rule. CLI runs must be deterministic, and bad input must exit with
code 2. Six deliberate bugs were planted, one at a time, to check that the tests catch them: no
pseudo-error, out-of-fold leakage, a flipped probit direction, split ties, a bootstrap that does not
resample, and an untempered out-of-fold arm. All six turned the suite red.

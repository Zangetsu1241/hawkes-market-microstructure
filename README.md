# Hawkes Processes for High-Frequency Trading Events

A research-grade bivariate, exponential-kernel Hawkes process to analyze the clustering of buy and sell trading events. Compares **self-exciting event intensities** against independently fitted homogeneous Poisson processes using strictly chronological held-out event data.

## Model

For type i in {buy,sell}:

`lambda_i(t) = mu_i + sum_j alpha[i,j] sum_{events k of type j before t} exp(-beta*(t-t_k))`

- `mu_i`: positive baseline arrival intensity.
- `alpha[i,j]`: excitation of target type i by a source event j.
- `beta`: fixed positive shared exponential decay.
- `alpha/beta`: branching matrix, spectral radius below 1 for stationarity.

Fit by constrained maximum likelihood with exact exponential-integral compensator, two initialization points, a column-sum stability bound and SciPy L-BFGS-B. Simulate exactly using Ogata thinning.

## Installation

```bash
python -m pip install -e '.[test]'
python -m pytest -q
hawkes-research --horizon 1000 --seed 42 --output results
```

For external trade event data, prepare a CSV with `time,side`, where `time` is a strictly increasing event timestamp measured in **seconds since the start of the observation**, and `side` is `buy` or `sell`:

```csv
time,side
0.1,buy
0.3,sell
0.35,buy
```

Then run:

```bash
hawkes-research --csv events.csv --horizon 3600 --beta 2 --output market-results
```

The observation horizon must exceed the last event timestamp. The input should contain enough events in both train and test intervals. Do not provide calendar dates or event timestamps in milliseconds without converting them to relative seconds.

## Research protocol

- First 70% of clock time: train; last 30%: held-out test.
- Hawkes parameters and Poisson intensities fit using **training events only**.
- At test start, carry forward the train-estimated Hawkes excitation state. During the test, calculate **prequential conditional likelihood**: each event is scored using only events already observed at that instant. Update history afterward.
- Compare held-out continuous-time point-process log likelihood and mark negative log likelihood. These are statistical goodness-of-fit measures, **not trading profits**.
- Compute pooled time-rescaling increments and a descriptive Kolmogorov-Smirnov diagnostic.

Outputs: `metrics.json`, `heldout_event_intensities.csv`, `diagnostics.png`.

## Reproducible synthetic experiment

Command: `hawkes-research --horizon 1000 --seed 42 --output results`.

| Metric | Hawkes | Independent Poisson |
|---|---:|---:|
| Test log likelihood | -529.445 | -594.380 |
| Log likelihood per test event | -0.8737 | -0.9808 |
| Test mark negative log likelihood | 0.6659 | 0.6818 |

Train events: 1,510. Test events: 606. Estimated Hawkes branching ratio: 0.5614. Improvement in held-out log likelihood: **+64.935**.

**This is synthetic data generated from a Hawkes process.** Better Hawkes fit is expected in such a setting and does not establish an advantage on real market order flows.

## Tests

12 pytest tests exercise seeded simulation, a Poisson special case, hand-calculated likelihood, train/test likelihood continuation, time-rescaling, stationarity, fitting, held-out model comparison, future-data invariance, validation, and CLI results. CI runs on Python 3.10 and 3.12.

## Limitations

The project models buy/sell arrival events, not quote updates, cancellations, fill probabilities or executable prices. It has fixed beta rather than learning a decay distribution, assumes no tied timestamps, and excludes overnight gaps and timestamp aggregation artifacts. Intraday seasonality and market opening/closing patterns can confound self-excitation findings. Real-world evaluation needs a point-in-time event-level dataset, realistic preprocessing, day-by-day testing and comparisons to time-varying Poisson or other richer processes. This is research software, not a trading recommendation.

MIT licensed.

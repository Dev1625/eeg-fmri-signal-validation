# EEG/fMRI signal validation

How do you tell if an EEG cleaning method removed heartbeat artifact while preserving the neural signal you wanted to measure? This repo puts both measurements next to a simple downstream test of how well EEG alpha power predicts a simulated fMRI BOLD signal.

The original research was an independent collaboration with a Meta AI researcher. I benchmarked four corrections on a 20-participant simultaneous EEG/fMRI dataset. Two hybrid methods removed 96.2% and 96.9% of heartbeat artifact while distorting the alpha rhythm. I stopped the voxelwise analysis after the scan timing remained ambiguous and a signal recovery check missed its thresholds. See the [research notes](docs/research-notes.md) for the stopping decision and dataset citation.

## Run it

Use Python 3.11 or newer. From the repo root, activate a virtual environment and run:

```sh
python -m pip install -r requirements.txt
python -m signal_validation
python -m unittest discover -s tests -v
```

The default run generates all inputs in memory and prints JSON. Use `python -m signal_validation --seed 17 --shifts 40` to change the seed and run fewer controls.

## Walkthrough

1. Generate a 10 Hz alpha rhythm and short heartbeat-like pulses whose spectra overlap. Keep the clean rhythm and injected artifact as separate arrays so we can measure both after correction.
2. Compare the raw signal, an imperfect template subtraction, and template subtraction followed by an 8–12 Hz notch. Extract alpha-band power in two-second windows and convolve it with a simplified hemodynamic response.
3. Predict BOLD with ridge regression. Hold out five contiguous test blocks, subtract a three-volume buffer from the training set around each block, and fit scaling on training data. Compare against a task-only baseline and 200 circular shifts of the EEG features.

The simulator and model live in [`signal_validation/benchmark.py`](signal_validation/benchmark.py). [`signal_validation/evaluation.py`](signal_validation/evaluation.py) adapts the blocked-score metrics and circular-shift logic from the original research code. AI assistance was used to prepare this public walkthrough and its tests.

## Synthetic results

Default run, seed 42. The task-only model scores **0.467** in held-out blocks.

| EEG input | Injected artifact energy removed | Alpha power vs. clean | Held-out R² | 95th percentile shift-null R² |
| --- | ---: | ---: | ---: | ---: |
| Raw | 0.0% | 205.0% | 0.504 | 0.468 |
| Template subtraction | 97.4% | 102.7% | 0.942 | 0.480 |
| Template + alpha notch | 99.1% | 0.1% | 0.465 | 0.470 |

The notch removes the most injected artifact and almost all alpha power. Its held-out score falls to the task-only baseline. The simulation gives the model a deliberately learnable EEG-to-BOLD relationship and a known artifact reference. The blocked scores quantify performance on this generated series. Generalization to new participants and the quality of real artifact references need separate evaluation.

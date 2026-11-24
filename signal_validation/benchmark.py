"""Compare alpha preservation and run a blocked EEG-to-BOLD baseline."""

from math import factorial

import numpy as np

from .evaluation import compute_cv_r2, compute_r, generate_time_shifted_features


FS = 128
TR = 2
VOLUMES = 240
MIN_SHIFT = 12
MAX_SHIFTS = VOLUMES - 2 * MIN_SHIFT + 1


def hemodynamic_response(values):
    t = np.arange(0, 32, TR)
    peak = t ** 5 * np.exp(-t) / factorial(5)
    undershoot = t ** 15 * np.exp(-t) / factorial(15)
    kernel = peak - undershoot / 6
    kernel /= kernel.sum()
    return np.convolve(values, kernel)[:len(values)]


def alpha_power(windows):
    n = windows.shape[1]
    window = np.hanning(n)
    hz = np.fft.rfftfreq(n, d=1 / FS)
    spectrum = np.abs(np.fft.rfft(windows * window, axis=1)) ** 2
    alpha = (hz >= 8) & (hz <= 12)
    return spectrum[:, alpha].sum(axis=1) * 2 / (n * np.sum(window ** 2))


def notch_alpha(windows):
    n = windows.shape[1]
    hz = np.fft.rfftfreq(n, d=1 / FS)
    spectrum = np.fft.rfft(windows, axis=1)
    spectrum[:, (hz >= 8) & (hz <= 12)] = 0
    return np.fft.irfft(spectrum, n=n, axis=1)


def simulate(seed=42, volumes=VOLUMES):
    rng = np.random.default_rng(seed)
    n = FS * TR
    t = np.arange(volumes * n) / FS
    condition = (np.arange(volumes) // 12) % 2
    slow = np.convolve(rng.normal(size=volumes), np.ones(5) / 5, mode='same')
    envelope = 1.2 + 0.3 * condition + 0.48 * slow
    clean = np.repeat(envelope, n) * np.sin(2 * np.pi * 10 * t)
    clean += 0.22 * rng.normal(size=t.size)

    # A brief pulse at each beat with power overlapping the alpha rhythm.
    phase = (t + 0.32) % (60 / 73) - 0.25
    artifact = 5 * np.exp(-0.5 * (phase / 0.045) ** 2) * np.cos(2 * np.pi * 10 * phase)
    artifact *= 1 + 0.7 * np.sin(2 * np.pi * t / 35)
    clean = clean.reshape(volumes, n)
    artifact = artifact.reshape(volumes, n)

    eeg = hemodynamic_response(alpha_power(clean))
    task = hemodynamic_response(condition)
    y = 0.8 * eeg + 0.23 * task + rng.normal(0, 0.08, volumes)
    return clean, artifact, task, y


def blocked_folds(n, folds=5, gap=3):
    if not 2 <= folds <= n or gap < 0:
        raise ValueError('Choose at least two nonempty folds and a nonnegative gap.')
    idx = np.arange(n)
    for test in np.array_split(idx, folds):
        train = idx[(idx < test[0] - gap) | (idx > test[-1] + gap)]
        if train.size == 0:
            raise ValueError('The gap leaves a fold with no training samples.')
        yield train, test


def ridge_predict(train_x, train_y, test_x, penalty=1.0):
    avg_x = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1
    train_z = (train_x - avg_x) / scale
    test_z = (test_x - avg_x) / scale
    avg_y = train_y.mean()
    system = train_z.T @ train_z + penalty * np.eye(train_z.shape[1])
    weights = np.linalg.solve(system, train_z.T @ (train_y - avg_y))
    return test_z @ weights + avg_y


def score(x, y):
    fold_scores = []
    for train, test in blocked_folds(len(y)):
        pred = ridge_predict(x[train], y[train], x[test])
        fold_scores.append((compute_cv_r2(y[test], pred, y[train].mean()), compute_r(y[test], pred)))
    r2, r = np.mean(fold_scores, axis=0)
    return float(r2), float(r)


def run(seed=42, n_shifts=200):
    if not 1 <= n_shifts <= MAX_SHIFTS:
        raise ValueError(f'Choose between 1 and {MAX_SHIFTS} shifts.')
    clean, artifact, task, y = simulate(seed)
    noisy = clean + artifact
    template = 0.84 * artifact
    signal = noisy - template
    methods = {
        'raw': (noisy, artifact),
        'template': (signal, artifact - template),
        'template + alpha notch': (notch_alpha(signal), notch_alpha(artifact - template)),
    }
    task_r2, _ = score(task[:, None], y)
    clean_alpha = np.mean(alpha_power(clean))
    artifact_energy = np.sum(artifact ** 2)
    offsets = np.random.default_rng(seed + 1).choice(
        np.arange(MIN_SHIFT, len(y) - MIN_SHIFT + 1), size=n_shifts, replace=False
    )
    report = []
    for name, (eeg, residual_artifact) in methods.items():
        power = alpha_power(eeg)
        x = hemodynamic_response(power)
        r2, r = score(np.column_stack([task, x]), y)
        null = []
        for k in offsets:
            shifted = generate_time_shifted_features(x, k)
            null.append(score(np.column_stack([task, shifted]), y)[0])
        report.append({
            'method': name,
            'artifact_removed_pct': float(100 * (1 - np.sum(residual_artifact ** 2) / artifact_energy)),
            'alpha_vs_clean_pct': float(100 * np.mean(power) / clean_alpha),
            'cv_r2': r2,
            'cv_r': r,
            'shift_null_95th_r2': float(np.quantile(null, 0.95)),
        })
    return {'seed': seed, 'shifts': n_shifts, 'task_only_cv_r2': task_r2, 'methods': report}

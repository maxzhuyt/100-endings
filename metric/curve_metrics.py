"""Story-level statistics of a 100-Endings no-rate curve (paper Section 3.1, Appendix C.6).

A curve is the sequence of position-level no-rates at the judged positions
(10% <= pct revealed <= 99%), sorted by pct.
  mean no-rate      mean over all judged positions
  late no-rate      mean over positions with pct >= 80
  retention         for every local peak (higher than both neighbours) on the raw curve,
                    min(no-rate over the next 10 positions) / peak; story value = mean over peaks
  inflection rate   share of positions where the smoothed curve (moving average, w=3)
                    reverses direction with vertex angle <= alpha degrees, after rescaling
                    x (pct) and y (no-rate) of the story onto the unit square
Logic is identical to the notebook that produced the paper tables
(creative-writing-bench/ending_results_analysis/notebooks/analysis_judge_norate.ipynb).
"""
import numpy as np

SMOOTH_W = 3


def smooth(y, w=SMOOTH_W):
    y = np.asarray(y, float)
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def inflection_rate(no_rate, pct, max_angle_deg=60):
    y = smooth(no_rate)
    n = len(y)
    if n < 5:
        return np.nan
    pct = np.asarray(pct, float)
    xn = (pct - pct.min()) / (pct.max() - pct.min() + 1e-12)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    count = 0
    for i in range(1, n - 1):
        d1, d2 = yn[i] - yn[i - 1], yn[i + 1] - yn[i]
        if d1 * d2 >= 0:
            continue
        v1 = np.array([xn[i - 1] - xn[i], yn[i - 1] - yn[i]])
        v2 = np.array([xn[i + 1] - xn[i], yn[i + 1] - yn[i]])
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        if np.degrees(np.arccos(np.clip(cos_a, -1, 1))) <= max_angle_deg:
            count += 1
    return count / n


def retention(no_rate, window=10):
    y = np.asarray(no_rate, float)
    n = len(y)
    if n < 3:
        return np.nan
    ratios = []
    for i in range(1, n - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            future = y[i + 1: min(i + 1 + window, n)]
            if len(future) and y[i] > 0:
                ratios.append(float(np.min(future)) / float(y[i]))
    return float(np.mean(ratios)) if ratios else np.nan


def story_stats(pct, no_rate):
    pct = np.asarray(pct, float)
    no_rate = np.asarray(no_rate, float)
    o = np.argsort(pct)
    pct, no_rate = pct[o], no_rate[o]
    late = no_rate[pct >= 80]
    return {
        "mean_nr": float(no_rate.mean()),
        "late_nr": float(late.mean()) if len(late) else np.nan,
        "retain_pct": retention(no_rate) * 100,
        "ir30": inflection_rate(no_rate, pct, 30),
        "ir60": inflection_rate(no_rate, pct, 60),
        "ir120": inflection_rate(no_rate, pct, 120),
    }

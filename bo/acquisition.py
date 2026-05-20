"""Acquisition functions for Bayesian Optimization."""
import numpy as np
from scipy.stats import norm


def UCB(preds, uncert, h_param=0.2, **kwargs):
    return preds + h_param * uncert


def LCB(preds, uncert, h_param=0.01, **kwargs):
    return preds - h_param * uncert


def EI(preds, uncert, h_param=0.01, **kwargs):
    improves = preds - kwargs["best_val"] - h_param
    z = improves / uncert
    return improves * norm.cdf(z) + uncert * norm.pdf(z)


def PI(preds, uncert, h_param=0.01, **kwargs):
    return norm.cdf(
        (preds - kwargs["best_val"] - h_param) / uncert
    )


def TS(preds, uncert, h_param=0.0, **kwargs):
    return kwargs["rng"].normal(preds, uncert)


def NEI(preds, uncert, h_param=0.01, **kwargs):
    """Noisy Expected Improvement: robust to observation noise.

    Accounts for observation noise variance (noise_var) in the improvement
    calculation, making it more stable when fitness measurements are noisy
    (e.g. experimental binding assays, MD-derived scores).
    """
    noise_var = kwargs.get("noise_var", 0.0)
    total_var = uncert ** 2 + noise_var
    total_std = np.sqrt(total_var)
    improves = preds - kwargs["best_val"] - h_param
    z = improves / total_std
    return improves * norm.cdf(z) + total_std * norm.pdf(z)


def QUCB(preds, uncert, h_param=0.2, **kwargs):
    """Quadratic UCB: penalizes high-uncertainty regions more aggressively.

    Unlike linear UCB, the uncertainty term is quadratic, which discourages
    the optimizer from venturing into poorly-modeled regions. Useful when
    the GP surrogate is unreliable far from training data.
    """
    return preds + h_param * uncert ** 2


def Greedy(preds, uncert, h_param=0.0, **kwargs):
    return preds

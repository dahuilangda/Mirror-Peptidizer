"""Bayesian Optimization module for D-peptide sequence optimization (Tier 2).

Provides Evolutionary BO and MCMC explorers with GP surrogate models,
supporting one-hot, physicochemical, and Boltz2 sequence embeddings.
"""
from .encoders import OneHotEncoder, PhysicochemicalEncoder, Boltz2Encoder, AAS
from .models import GPRegressor
from .explorers import BO_EVO, MCMC, Boltz2BO
from .landscape import EXPLandscape
from .scoring import FuzzyScore, swi, swi_weights
from .acquisition import UCB, LCB, EI, PI, TS, Greedy, NEI, QUCB

__all__ = [
    "OneHotEncoder", "PhysicochemicalEncoder", "Boltz2Encoder", "AAS",
    "GPRegressor",
    "BO_EVO", "MCMC", "Boltz2BO",
    "EXPLandscape",
    "FuzzyScore", "swi", "swi_weights",
    "UCB", "LCB", "EI", "PI", "TS", "Greedy", "NEI", "QUCB",
]

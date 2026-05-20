"""Scoring utilities: FuzzyScore, SWI solubility, and sequence helpers."""
import contextlib
from typing import List, Sequence as SeqType, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ---------- sequence utilities ----------

AAS = "ILVAGMFYWEDQNHCRKSTP"


def string_to_one_hot(sequence, alphabet):
    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out


def one_hot_to_string(one_hot, alphabet):
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])


def construct_mutant_from_sample(pwm_sample, one_hot_base):
    one_hot = np.zeros(one_hot_base.shape)
    one_hot += one_hot_base
    i, j = np.nonzero(pwm_sample)
    one_hot[i, :] = 0
    one_hot[i, j] = 1
    return one_hot


# ---------- SWI solubility ----------

swi_weights = {
    'A': 0.8356471476582918, 'C': 0.5208088354857734,
    'E': 0.9876987431418378, 'D': 0.9079044671339564,
    'G': 0.7997168496420723, 'F': 0.5849790194237692,
    'I': 0.6784124413866582, 'H': 0.8947913996466419,
    'K': 0.9267104557513497, 'M': 0.6296623675420369,
    'L': 0.6554221515081433, 'N': 0.8597433107431216,
    'Q': 0.789434648348208,  'P': 0.8235328714705341,
    'S': 0.7440908318492778, 'R': 0.7712466317693457,
    'T': 0.8096922697856334, 'W': 0.6374678690957594,
    'V': 0.7357837119163659, 'Y': 0.6112801822947587,
}


def swi(seq, weights=None):
    """Solubility Weighted Index: mean of per-residue solubility weights."""
    if weights is None:
        weights = swi_weights
    return np.mean([weights[aa] for aa in seq])


def _longest_run(seq, predicate):
    longest = 0
    current = 0
    for aa in seq:
        if predicate(aa):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


_HYDROPHOBIC = set('AILMFWYV')
_AROMATIC = set('FWY')
_CHARGED = set('DEKRH')
_BETA_BRANCHED = set('IVT')
_OXIDATION_LABILE = set('CMW')
_ASPARTIMIDE_FOLLOWERS = set('GSTNDA')
_DEAMIDATION_FOLLOWERS = set('GS')

_KD_HYDROPATHY = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
    'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
    'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
    'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5,
}

# Chou-Fasman beta-sheet propensities, used here only as a local structure
# proxy for protected resin-bound peptide packing risk.
_BETA_SHEET_PROPENSITY = {
    'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
    'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
    'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
    'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47,
}

# Literature-informed aggregation propensity for protected, resin-bound chains.
# Scale is centered near 1.0: values above 1.0 are aggregation-prone under
# standard Fmoc-SPPS; values below 1.0 are comparatively disruptive/solvating.
_SPPS_AGGREGATION_PROPENSITY = {
    'A': 0.76, 'C': 1.15, 'D': 0.62, 'E': 0.58, 'F': 1.30,
    'G': 0.64, 'H': 0.85, 'I': 1.53, 'K': 0.62, 'L': 1.20,
    'M': 0.92, 'N': 1.10, 'P': 0.46, 'Q': 1.00, 'R': 1.05,
    'S': 0.76, 'T': 0.96, 'V': 1.36, 'W': 1.18, 'Y': 1.08,
}

# Relative process difficulty for adding each incoming Fmoc-amino acid.
# The values are dimensionless engineering risk weights, not predicted yield.
_INCOMING_COUPLING_LOAD = {
    'G': 0.03, 'A': 0.05, 'S': 0.08, 'D': 0.10, 'E': 0.10,
    'L': 0.12, 'N': 0.14, 'Q': 0.14, 'K': 0.13, 'P': 0.15,
    'M': 0.16, 'H': 0.16, 'F': 0.16, 'Y': 0.18, 'R': 0.20,
    'V': 0.22, 'C': 0.22, 'W': 0.24, 'I': 0.25, 'T': 0.18,
}


def _fraction(seq, residues):
    return sum(aa in residues for aa in seq) / len(seq)


def _net_charge_ph7(seq):
    # Approximate side-chain charge at neutral pH; terminal chemistry is
    # intentionally omitted because the benchmark does not encode capping.
    return (
        seq.count('K') + seq.count('R') + 0.1 * seq.count('H')
        - seq.count('D') - seq.count('E')
    )


def _gravy(seq):
    return float(np.mean([_KD_HYDROPATHY[aa] for aa in seq]))


def _sliding_window_scores(seq, values, window=5):
    if len(seq) < window:
        return [float(np.mean([values[aa] for aa in seq]))]
    return [
        float(np.mean([values[aa] for aa in seq[i:i + window]]))
        for i in range(len(seq) - window + 1)
    ]


def _coupling_step_risk(incoming, acceptor):
    risk = _INCOMING_COUPLING_LOAD[incoming]

    # Coupling onto an N-terminal Pro on resin is a known difficult X-Pro step.
    if acceptor == 'P':
        risk += 0.34
    elif acceptor in _BETA_BRANCHED:
        risk += 0.12
    elif acceptor in _AROMATIC:
        risk += 0.06

    if incoming in _BETA_BRANCHED and acceptor in _BETA_BRANCHED.union({'P'}):
        risk += 0.16
    if incoming in _AROMATIC and acceptor in _HYDROPHOBIC:
        risk += 0.10
    if incoming == 'P' and acceptor == 'P':
        risk += 0.16
    if incoming == 'R':
        risk += 0.06
    if incoming == 'C':
        risk += 0.05
    return risk


def _fragment_aggregation_risk(fragment):
    if len(fragment) < 6:
        return 0.0

    hydrophobic_fraction = _fraction(fragment, _HYDROPHOBIC)
    aromatic_fraction = _fraction(fragment, _AROMATIC)
    charge_density = abs(_net_charge_ph7(fragment)) / len(fragment)
    gravy_norm = (_gravy(fragment) + 4.5) / 9.0
    protected_pa = float(np.mean([_SPPS_AGGREGATION_PROPENSITY[aa] for aa in fragment]))
    beta_sheet = float(np.mean([_BETA_SHEET_PROPENSITY[aa] for aa in fragment]))
    max_hydrophobic_run = _longest_run(fragment, lambda aa: aa in _HYDROPHOBIC)
    max_beta_run = _longest_run(fragment, lambda aa: aa in _BETA_BRANCHED.union(_AROMATIC))

    risk = 0.0
    risk += max(0.0, hydrophobic_fraction - 0.42) * 1.25
    risk += max(0.0, aromatic_fraction - 0.16) * 0.90
    risk += max(0.0, gravy_norm - 0.56) * 0.85
    risk += max(0.0, protected_pa - 0.95) * 0.95
    risk += max(0.0, beta_sheet - 1.08) * 0.50
    risk += max(0.0, 0.16 - charge_density) * 0.55
    risk += max(0, max_hydrophobic_run - 4) * 0.15
    risk += max(0, max_beta_run - 3) * 0.10
    risk *= min(1.0, len(fragment) / 12.0)
    risk += max(0, len(fragment) - 18) * 0.015
    return float(risk)


def _spps_cycle_metrics(seq):
    step_risks = []
    high_risk_steps = 0
    max_consecutive_difficult = 0
    consecutive_difficult = 0

    # Fmoc SPPS grows from the C-terminus to the N-terminus. At each cycle,
    # incoming seq[i] is coupled onto the current N-terminal residue seq[i + 1].
    for i in range(len(seq) - 2, -1, -1):
        risk = _coupling_step_risk(seq[i], seq[i + 1])
        step_risks.append(risk)
        if risk >= 0.45:
            high_risk_steps += 1
            consecutive_difficult += 1
        else:
            consecutive_difficult = 0
        max_consecutive_difficult = max(max_consecutive_difficult, consecutive_difficult)

    if step_risks:
        coupling_mean = float(np.mean(step_risks))
        coupling_max = float(np.max(step_risks))
        coupling_p90 = float(np.percentile(step_risks, 90))
    else:
        coupling_mean = coupling_max = coupling_p90 = 0.0

    coupling_risk = (
        coupling_mean
        + max(0.0, coupling_max - 0.40) * 0.45
        + max(0.0, coupling_p90 - 0.35) * 0.30
        + high_risk_steps * 0.075
        + max(0, max_consecutive_difficult - 1) * 0.12
    )

    return {
        'spps_coupling_mean_risk': coupling_mean,
        'spps_coupling_max_risk': coupling_max,
        'spps_coupling_p90_risk': coupling_p90,
        'spps_high_risk_couplings': high_risk_steps,
        'spps_max_consecutive_difficult_couplings': max_consecutive_difficult,
        'spps_coupling_risk': float(coupling_risk),
    }


def _aggregation_metrics(seq):
    fragment_risks = []
    onset_cycle = 0
    for i in range(len(seq) - 1, -1, -1):
        fragment = seq[i:]
        risk = _fragment_aggregation_risk(fragment)
        fragment_risks.append(risk)
        if onset_cycle == 0 and risk >= 0.75:
            onset_cycle = len(seq) - i

    pa_windows = _sliding_window_scores(seq, _SPPS_AGGREGATION_PROPENSITY, window=5)
    beta_windows = _sliding_window_scores(seq, _BETA_SHEET_PROPENSITY, window=5)
    high_pa_windows = sum(score >= 1.10 for score in pa_windows)
    high_beta_windows = sum(score >= 1.20 for score in beta_windows)
    max_consecutive_high_pa = 0
    consecutive = 0
    for score in pa_windows:
        if score >= 1.10:
            consecutive += 1
        else:
            consecutive = 0
        max_consecutive_high_pa = max(max_consecutive_high_pa, consecutive)

    c_terminal_window = seq[-min(8, len(seq)):]
    c_terminal_pa = float(np.mean([_SPPS_AGGREGATION_PROPENSITY[aa] for aa in c_terminal_window]))
    c_terminal_beta = float(np.mean([_BETA_SHEET_PROPENSITY[aa] for aa in c_terminal_window]))
    max_risk = float(max(fragment_risks)) if fragment_risks else 0.0
    mean_risk = float(np.mean(fragment_risks)) if fragment_risks else 0.0
    risk_area = float(np.sum(fragment_risks) / max(1, len(seq)))
    early_onset_penalty = max(0.0, (len(seq) * 0.60 - onset_cycle) / len(seq)) if onset_cycle else 0.0
    local_nucleus_risk = (
        max(0.0, max(pa_windows) - 1.05) * 0.85
        + max(0.0, max(beta_windows) - 1.18) * 0.45
        + max(0, max_consecutive_high_pa - 1) * 0.20
        + high_pa_windows * 0.045
        + high_beta_windows * 0.035
    )
    c_terminal_risk = (
        max(0.0, c_terminal_pa - 0.95) * 0.55
        + max(0.0, c_terminal_beta - 1.05) * 0.30
    )
    aggregation_risk = (
        0.50 * max_risk
        + 0.28 * risk_area
        + 0.24 * local_nucleus_risk
        + 0.18 * c_terminal_risk
        + 0.20 * early_onset_penalty
    )
    return {
        'spps_aggregation_max_risk': max_risk,
        'spps_aggregation_mean_risk': mean_risk,
        'spps_aggregation_risk_area': risk_area,
        'spps_aggregation_onset_cycle': onset_cycle,
        'spps_pa_max_5mer': float(max(pa_windows)),
        'spps_pa_mean_5mer': float(np.mean(pa_windows)),
        'spps_beta_sheet_max_5mer': float(max(beta_windows)),
        'spps_beta_sheet_mean_5mer': float(np.mean(beta_windows)),
        'spps_high_pa_5mer_count': high_pa_windows,
        'spps_high_beta_5mer_count': high_beta_windows,
        'spps_max_consecutive_high_pa_5mer': max_consecutive_high_pa,
        'spps_c_terminal_pa': c_terminal_pa,
        'spps_c_terminal_beta_sheet': c_terminal_beta,
        'spps_local_nucleus_risk': float(local_nucleus_risk),
        'spps_c_terminal_aggregation_risk': float(c_terminal_risk),
        'spps_aggregation_risk': float(aggregation_risk),
    }


def _side_reaction_metrics(seq):
    aspartimide_motifs = 0
    deamidation_motifs = 0
    acid_labile_pairs = 0
    for i in range(len(seq) - 1):
        pair = seq[i:i + 2]
        if pair[0] == 'D' and pair[1] in _ASPARTIMIDE_FOLLOWERS:
            aspartimide_motifs += 1
        if pair[0] in {'N', 'Q'} and pair[1] in _DEAMIDATION_FOLLOWERS:
            deamidation_motifs += 1
        if pair in {'DP', 'NP'}:
            acid_labile_pairs += 1

    cys_count = seq.count('C')
    met_count = seq.count('M')
    trp_count = seq.count('W')
    his_count = seq.count('H')

    oxidation_liability = (
        0.22 * cys_count
        + 0.08 * met_count
        + 0.10 * trp_count
        + (0.35 if cys_count >= 2 else 0.0)
    )
    protecting_group_load = (
        0.06 * (seq.count('R') + seq.count('H'))
        + 0.05 * (seq.count('C') + seq.count('W'))
        + 0.03 * (seq.count('N') + seq.count('Q') + seq.count('Y'))
    )
    side_reaction_risk = (
        0.38 * aspartimide_motifs
        + 0.16 * deamidation_motifs
        + 0.12 * acid_labile_pairs
        + oxidation_liability
        + protecting_group_load
        + (0.12 if seq[0] in {'Q', 'E'} else 0.0)
    )
    return {
        'aspartimide_motif_count': aspartimide_motifs,
        'deamidation_motif_count': deamidation_motifs,
        'acid_labile_pair_count': acid_labile_pairs,
        'cys_count': cys_count,
        'met_count': met_count,
        'trp_count': trp_count,
        'his_count': his_count,
        'oxidation_liability': float(oxidation_liability),
        'protecting_group_load': float(protecting_group_load),
        'side_reaction_risk': float(side_reaction_risk),
    }


def _purification_metrics(seq):
    length = len(seq)
    hydrophobic_fraction = _fraction(seq, _HYDROPHOBIC)
    aromatic_fraction = _fraction(seq, _AROMATIC)
    charged_fraction = _fraction(seq, _CHARGED)
    net_charge = _net_charge_ph7(seq)
    charge_density = abs(net_charge) / length
    gravy = _gravy(seq)
    max_hydrophobic_run = _longest_run(seq, lambda aa: aa in _HYDROPHOBIC)
    max_repeat_run = max(_longest_run(seq, lambda aa, target=target: aa == target) for target in set(seq))

    retention_index = (
        0.60 * hydrophobic_fraction
        + 0.25 * aromatic_fraction
        + 0.12 * max(0.0, gravy)
        + 0.02 * max(0, length - 15)
        - 0.22 * charged_fraction
    )
    purification_risk = 0.0
    purification_risk += max(0.0, retention_index - 0.42) * 1.25
    purification_risk += max(0.0, 0.12 - charge_density) * 0.80
    purification_risk += max(0, max_hydrophobic_run - 5) * 0.10
    purification_risk += max(0, max_repeat_run - 4) * 0.06

    return {
        'SWI': swi(seq),
        'hydrophobic_fraction': hydrophobic_fraction,
        'aromatic_fraction': aromatic_fraction,
        'charged_fraction': charged_fraction,
        'net_charge_ph7': float(net_charge),
        'charge_density_ph7': float(charge_density),
        'gravy': float(gravy),
        'max_hydrophobic_run': max_hydrophobic_run,
        'max_repeat_run': max_repeat_run,
        'rp_hplc_retention_index': float(retention_index),
        'purification_risk': float(purification_risk),
    }


def synthesis_metrics(seq):
    """SPPS synthesis-risk metrics for BO penalization.

    The model is deterministic and sequence-based. It scores the same C-to-N
    construction path used in Fmoc-SPPS, combining incoming/acceptor coupling
    difficulty, resin-bound aggregation windows, side-reaction liabilities, and
    purification risk. It is an auditable process-risk objective for screening,
    not a substitute for route scouting or measured crude purity.
    """
    seq = str(seq).strip().upper()
    invalid = sorted(set(seq) - set(AAS))
    if not seq or invalid:
        raise ValueError(f'Invalid peptide sequence for synthesis metrics: {seq}')

    cycle = _spps_cycle_metrics(seq)
    aggregation = _aggregation_metrics(seq)
    side = _side_reaction_metrics(seq)
    purification = _purification_metrics(seq)

    length_risk = max(0, len(seq) - 24) * 0.035 + max(0, len(seq) - 35) * 0.065
    synthesis_penalty = (
        1.00 * cycle['spps_coupling_risk']
        + 1.15 * aggregation['spps_aggregation_risk']
        + 0.95 * side['side_reaction_risk']
        + 0.80 * purification['purification_risk']
        + 0.60 * length_risk
    )
    feasibility = 100.0 * np.exp(-synthesis_penalty / 3.0)
    if synthesis_penalty < 0.8:
        risk_class = 'low'
    elif synthesis_penalty < 1.6:
        risk_class = 'moderate'
    elif synthesis_penalty < 2.6:
        risk_class = 'high'
    else:
        risk_class = 'very_high'

    return {
        'sequence_length': len(seq),
        **cycle,
        **aggregation,
        **side,
        **purification,
        'length_risk': float(length_risk),
        'synthesis_penalty': float(synthesis_penalty),
        'synthesis_feasibility_score': float(feasibility),
        'synthesis_risk_class': risk_class,
    }


# ---------- FuzzyScore ----------

class FuzzyScore:
    """Multi-objective desirability scoring using geometric mean.

    Each property has a desirability function defined by control points
    (x, y) that define a piecewise-linear curve. The final fuzzy score
    is the geometric mean of individual desirability scores.
    """

    def __init__(self, df):
        self.df = df.copy()

    @staticmethod
    def _create_desirability_function(desirability, truncate_left=True, truncate_right=True):
        x = [point['x'] for point in desirability]
        y = [point['y'] if point['y'] != 0 else 1e-9 for point in desirability]
        assert len(x) == len(y)
        if truncate_left:
            x = [x[0] - 1] + x
            y = [y[0]] + y
        if truncate_right:
            x.append(x[-1] + 1)
            y.append(y[-1])
        return interp1d(x, y, fill_value='extrapolate')

    def _score_func(self, desirability, weight, x):
        func = self._create_desirability_function(desirability)
        return func(x) + (1 - func(x)) * (1 - weight)

    def fuzzy_score(self, **kwargs):
        """Compute fuzzy score for all rows in the DataFrame.

        Args:
            kwargs: {property_name: (desirability_points, importance_weight)}
                desirability_points: list of {'x': float, 'y': float}
                importance_weight: float in [0, 1]

        Returns:
            DataFrame with 'FuzzyScore' column added.
        """
        fuzzy_list = []
        for idx in range(self.df.shape[0]):
            fuzzy, n = 1.0, 0
            score_list = []
            values = self.df.iloc[idx]
            for k, m in kwargs.items():
                if k in self.df.columns:
                    score = self._score_func(m[0], m[1], values[k])
                    score_list.append(score)
            for s in score_list:
                if s is not None and not np.isnan(s):
                    fuzzy *= s
                    n += 1
            fuzzy_list.append(fuzzy ** (1 / n) if n > 0 else fuzzy)

        self.df['FuzzyScore'] = fuzzy_list
        self.df.sort_values(by=['FuzzyScore'], ascending=False, inplace=True)
        return self.df

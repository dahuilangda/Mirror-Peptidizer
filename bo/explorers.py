"""Evolutionary Bayesian Optimization and MCMC explorers.

Ported from D_Peptide_BO/bo_evolutionary.py and mcmc_evolutionary.py,
with fasthit dependencies removed. Uses local Encoder/Model/Landscape interfaces.
"""
import json
import math
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from . import acquisition as acq
from .scoring import (
    AAS,
    construct_mutant_from_sample,
    one_hot_to_string,
    string_to_one_hot,
    synthesis_metrics,
)


def _hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def _random_mutant(sequence, alphabet, rng, min_mutations, max_mutations):
    seq = list(sequence)
    n_mut = int(rng.integers(min_mutations, max_mutations + 1))
    positions = rng.choice(len(seq), size=n_mut, replace=False)
    for pos in positions:
        choices = [aa for aa in alphabet if aa != seq[pos]]
        seq[pos] = choices[int(rng.integers(len(choices)))]
    return ''.join(seq)


def _count_diverse_feasible(sequences, min_hamming):
    selected = []
    for seq in sequences:
        if all(_hamming_distance(seq, kept) >= min_hamming for kept in selected):
            selected.append(seq)
    return len(selected)


def validate_bo_proposal_config(
    sequence,
    trials,
    model_queries,
    alphabet=AAS,
    min_mutations=1,
    max_mutations=None,
    batch_diversity_min_hamming=2,
    proposal_synthesis_penalty_weight=0.25,
    proposal_max_synthesis_penalty=3.1,
    preflight_samples=None,
):
    """Validate proposal-space capacity before a BO run starts."""
    seq_len = len(sequence)
    if seq_len == 0:
        raise ValueError("starting sequence is empty")
    if trials < 1:
        raise ValueError("trials must be positive")
    if model_queries < trials:
        raise ValueError("model_queries must be >= trials")
    if min_mutations < 1:
        raise ValueError("min_mutations must be >= 1")
    if max_mutations is None:
        max_mutations = min(3, seq_len)
    max_mutations = int(max_mutations)
    if max_mutations < min_mutations:
        raise ValueError("max_mutations must be >= min_mutations")
    if max_mutations > seq_len:
        raise ValueError("max_mutations cannot exceed sequence length")
    if batch_diversity_min_hamming < 1:
        raise ValueError("batch_diversity_min_hamming must be >= 1")
    if batch_diversity_min_hamming > max_mutations * 2:
        raise ValueError(
            "batch_diversity_min_hamming is too strict for the mutation radius; "
            "it cannot exceed 2 * max_mutations"
        )
    if proposal_synthesis_penalty_weight <= 0:
        raise ValueError("proposal_synthesis_penalty_weight must be > 0")
    if proposal_max_synthesis_penalty <= 0:
        raise ValueError("proposal_max_synthesis_penalty must be > 0")

    aa_options = max(0, len(alphabet) - 1)
    candidate_capacity = 0
    for k in range(int(min_mutations), max_mutations + 1):
        if k > seq_len:
            break
        candidate_capacity += math.comb(seq_len, k) * (aa_options ** k)
    screen_count = int(model_queries // trials)
    minimum_capacity = max(trials * 8, screen_count)
    if candidate_capacity < minimum_capacity:
        raise ValueError(
            f"proposal space has only {candidate_capacity} candidates; "
            f"need at least {minimum_capacity} for trials={trials}, "
            f"model_queries={model_queries}"
        )
    if preflight_samples is None:
        preflight_samples = max(int(model_queries), int(trials) * 200)
    rng = np.random.default_rng(0)
    feasible = []
    seen = set()
    max_draws = int(preflight_samples) * 3
    draws = 0
    while len(seen) < int(preflight_samples) and draws < max_draws:
        seq = _random_mutant(
            sequence,
            alphabet,
            rng,
            int(min_mutations),
            max_mutations,
        )
        seen.add(seq)
        draws += 1
    for seq in seen:
        if synthesis_metrics(seq)['synthesis_penalty'] <= proposal_max_synthesis_penalty:
            feasible.append(seq)
    diverse_feasible = _count_diverse_feasible(feasible, batch_diversity_min_hamming)
    if diverse_feasible < trials:
        raise ValueError(
            f"preflight found only {diverse_feasible} feasible diverse candidates; "
            f"need {trials}. Increase proposal_max_synthesis_penalty, "
            "increase max_mutations, or lower batch_diversity_min_hamming."
        )
    return {
        "max_mutations": max_mutations,
        "candidate_capacity": candidate_capacity,
        "screen_count": screen_count,
        "preflight_feasible": len(feasible),
        "preflight_diverse_feasible": diverse_feasible,
    }


class BO_EVO:
    """Synthesis-aware, ARD-guided batched Bayesian Optimization explorer."""

    def __init__(
        self,
        encoder,
        model,
        rounds: int,
        expmt_queries_per_round: int,
        model_queries_per_round: int,
        starting_sequence: str,
        alphabet: str = AAS,
        log_file: Optional[str] = None,
        seed: int = 0,
        util_func: str = "UCB",
        uf_param: float = 0.2,
        recomb_rate: float = 0.0,
        min_mutations: int = 1,
        max_mutations: Optional[int] = None,
        batch_diversity_min_hamming: int = 2,
        proposal_synthesis_penalty_weight: float = 0.25,
        proposal_max_synthesis_penalty: float = 3.1,
        preflight_samples: int = 600,
    ):
        if not hasattr(model, "uncertainties"):
            raise TypeError("BO_EVO requires a surrogate model exposing uncertainties")
        if expmt_queries_per_round < 1:
            raise ValueError("expmt_queries_per_round must be positive")
        if model_queries_per_round < expmt_queries_per_round:
            raise ValueError("model_queries_per_round must be >= expmt_queries_per_round")
        if batch_diversity_min_hamming < 1:
            raise ValueError("batch_diversity_min_hamming must be >= 1")
        if proposal_synthesis_penalty_weight <= 0:
            raise ValueError("proposal_synthesis_penalty_weight must be > 0")
        if proposal_max_synthesis_penalty <= 0:
            raise ValueError("proposal_max_synthesis_penalty must be > 0")
        checked = validate_bo_proposal_config(
            starting_sequence,
            expmt_queries_per_round,
            model_queries_per_round,
            alphabet=alphabet,
            min_mutations=min_mutations,
            max_mutations=max_mutations,
            batch_diversity_min_hamming=batch_diversity_min_hamming,
            proposal_synthesis_penalty_weight=proposal_synthesis_penalty_weight,
            proposal_max_synthesis_penalty=proposal_max_synthesis_penalty,
            preflight_samples=preflight_samples,
        )

        self._name = (
            f"BO_EVO_proposal-function={util_func}"
            f"_diversity-h{batch_diversity_min_hamming}"
            f"_synth-weight{proposal_synthesis_penalty_weight}"
        )

        self._encoder = encoder
        self._model = model
        self._rounds = rounds
        self._expmt_queries_per_round = expmt_queries_per_round
        self._model_queries_per_round = model_queries_per_round
        self._starting_sequence = starting_sequence
        self._log_file = log_file
        self._rng = np.random.default_rng(seed)
        self._alphabet = alphabet
        self._recomb_rate = recomb_rate
        self._min_mutations = max(1, int(min_mutations))
        self._max_mutations = checked["max_mutations"]
        self._batch_diversity_min_hamming = int(batch_diversity_min_hamming)
        self._proposal_synthesis_penalty_weight = float(proposal_synthesis_penalty_weight)
        self._proposal_max_synthesis_penalty = float(proposal_max_synthesis_penalty)
        self._best_fitness = 0.0
        self._state = string_to_one_hot(starting_sequence, alphabet)
        self._seq_len = len(starting_sequence)

        util_funcs = {
            "UCB": acq.UCB, "LCB": acq.LCB, "EI": acq.EI,
            "PI": acq.PI, "TS": acq.TS, "Greedy": acq.Greedy,
            "NEI": acq.NEI, "QUCB": acq.QUCB,
        }
        if util_func not in util_funcs:
            raise ValueError(f"Unsupported acquisition function: {util_func}")
        self._util_func = util_funcs[util_func]
        self._uf_param = uf_param

    def _recombine_population(self, gen):
        self._rng.shuffle(gen)
        ret = []
        for i in range(0, len(gen) - 1, 2):
            strA, strB = [], []
            switch = False
            for ind in range(len(gen[i])):
                if self._rng.random() < self._recomb_rate:
                    switch = not switch
                if switch:
                    strA.append(gen[i][ind])
                    strB.append(gen[i + 1][ind])
                else:
                    strB.append(gen[i][ind])
                    strA.append(gen[i + 1][ind])
            ret.append("".join(strA))
            ret.append("".join(strB))
        return ret

    def _position_uncertainties(self):
        """Estimate per-position sampling weights for mutation proposals."""
        if getattr(self._encoder, 'position_sampling', 'ard') == 'uniform':
            return np.full(self._seq_len, 1.0 / self._seq_len, dtype=float)
        if not hasattr(self._model, "ard_lengthscale"):
            raise RuntimeError("GP model does not expose ARD length scales")
        ls = np.asarray(self._model.ard_lengthscale, dtype=float)

        per_position_dim = getattr(self._encoder, 'per_position_dim', None)
        if per_position_dim is None:
            raise RuntimeError(
                f"Encoder {self._encoder.name} must define position_sampling "
                "or expose per_position_dim"
            )
        expected_dim = self._seq_len * int(per_position_dim)
        if len(ls) != expected_dim:
            raise RuntimeError(
                f"ARD length_scale dimension {len(ls)} does not match "
                f"sequence length {self._seq_len} x per_position_dim {per_position_dim}"
            )
        ls_per_pos = ls.reshape(self._seq_len, int(per_position_dim))
        relevance = 1.0 / (ls_per_pos.mean(axis=1) + 1e-8)
        total = relevance.sum()
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("Invalid ARD position relevance values")
        return relevance / total

    def _sample_actions(self, mask=None):
        actions = set()
        pos_changes = []
        current_residues = np.argmax(self._state, axis=1)
        for pos in range(self._seq_len):
            pos_changes.append([])
            for res in range(len(self._alphabet)):
                if res == current_residues[pos]:
                    continue
                if mask is None or not mask[res, pos]:
                    pos_changes[pos].append((pos, res))

        pos_changes_len = len([x for x in pos_changes if len(x) > 0])
        pos_changes_list = [pos for pos, x in enumerate(pos_changes) if len(x) > 0]
        if pos_changes_len == 0:
            raise RuntimeError("No mutable positions available")

        pos_weights = self._position_uncertainties()
        pos_prob = np.array([
            pos_weights[p] if len(pos_changes[p]) > 0 else 0.0
            for p in pos_changes_list
        ])
        pos_prob = pos_prob / pos_prob.sum()

        max_tries = 5000
        tries = 0
        target = int(self._model_queries_per_round / self._expmt_queries_per_round)
        while len(actions) < target:
            n_mut = max(self._min_mutations, self._rng.poisson(1))
            if self._max_mutations is not None:
                n_mut = min(n_mut, int(self._max_mutations))
            n_mut = min(n_mut, pos_changes_len)
            mut_positions = self._rng.choice(
                len(pos_changes_list), size=n_mut, replace=False,
                p=pos_prob,
            )
            action = []
            for idx in mut_positions:
                pos = pos_changes_list[idx]
                pos_tuple = pos_changes[pos][
                    self._rng.integers(len(pos_changes[pos]))
                ]
                action.append(pos_tuple)
            if len(action) > 0 and tuple(action) not in actions:
                actions.add(tuple(action))
            tries += 1
            if tries >= max_tries:
                break
        if len(actions) < target:
            raise RuntimeError(
                f"Only generated {len(actions)} unique actions, required {target}; "
                "increase model_queries_per_round or relax mutation constraints"
            )
        return list(actions)

    def _proposal_synthesis_penalties(self, sequences):
        penalties = []
        infeasible = []
        for seq in sequences:
            penalty = float(synthesis_metrics(seq)['synthesis_penalty'])
            penalties.append(penalty)
            infeasible.append(penalty > self._proposal_max_synthesis_penalty)
        return np.array(penalties, dtype=float), np.array(infeasible, dtype=bool)

    def _select_diverse_batch(self, samples, preds, rank_scores):
        if len(samples) == 0:
            raise RuntimeError("No unmeasured candidate sequences were proposed")

        finite = np.where(np.isfinite(rank_scores))[0]
        if len(finite) == 0:
            raise RuntimeError("No finite-ranked candidate sequences were proposed")
        jitter = self._rng.normal(0.0, 1e-9, size=len(rank_scores))
        order = np.argsort(rank_scores + jitter)[::-1]
        selected = []
        selected_preds = []
        selected_set = set()
        min_hamming = self._batch_diversity_min_hamming

        for idx in order:
            if not np.isfinite(rank_scores[idx]):
                continue
            seq = samples[idx]
            if seq in selected_set:
                continue
            if min_hamming > 0 and any(
                _hamming_distance(seq, kept) < min_hamming for kept in selected
            ):
                continue
            selected.append(seq)
            selected_set.add(seq)
            selected_preds.append(preds[idx])
            if len(selected) >= self._expmt_queries_per_round:
                break

        if len(selected) < self._expmt_queries_per_round:
            raise RuntimeError(
                f"Only {len(selected)} candidates satisfied batch diversity "
                f"Hamming >= {min_hamming}; required {self._expmt_queries_per_round}"
            )

        return np.array(selected), np.array(selected_preds)

    def _pick_action(self, selected_sequences=None, measured_sequences=None, mask=None):
        if selected_sequences is None:
            selected_sequences = []
        if measured_sequences is None:
            measured_sequences = set()
        else:
            measured_sequences = set(measured_sequences)
        state = self._state.copy()
        actions = self._sample_actions(mask=mask)
        if not actions:
            raise RuntimeError("No mutation actions available")
        actions_to_screen = []
        states_to_screen = []
        for i in range(len(actions)):
            x = np.zeros((self._seq_len, len(self._alphabet)))
            for action in actions[i]:
                x[action] = 1
            actions_to_screen.append(x)
            state_to_screen = construct_mutant_from_sample(x, state)
            states_to_screen.append(one_hot_to_string(state_to_screen, self._alphabet))

        penalties, infeasible = self._proposal_synthesis_penalties(states_to_screen)
        already_measured = np.array([seq in measured_sequences for seq in states_to_screen])
        excluded = infeasible | already_measured
        if selected_sequences:
            too_close = np.array([
                any(
                    _hamming_distance(seq, selected) < self._batch_diversity_min_hamming
                    for selected in selected_sequences
                )
                for seq in states_to_screen
            ])
            excluded = excluded | too_close

        candidate_idx = np.where(~excluded)[0]
        if len(candidate_idx) == 0:
            raise RuntimeError(
                "No candidate action satisfies synthesis, novelty, and batch diversity constraints"
            )

        candidate_sequences = [states_to_screen[i] for i in candidate_idx]
        encodings = self._encoder.encode(candidate_sequences)
        preds = self._model.get_fitness(encodings)

        kwargs = {"best_val": self._best_fitness, "rng": self._rng}
        utility = self._util_func(
            preds, self._model.uncertainties, h_param=self._uf_param, **kwargs
        )
        rank_scores = utility - self._proposal_synthesis_penalty_weight * penalties[candidate_idx]
        if not np.isfinite(rank_scores).any():
            raise RuntimeError("No finite-ranked candidate sequences were proposed")

        local_idx = int(np.argmax(rank_scores))
        action_idx = int(candidate_idx[local_idx])
        uncertainty = self._model.uncertainties[local_idx]
        action = actions_to_screen[action_idx]
        new_state_string = states_to_screen[action_idx]
        self._state = string_to_one_hot(new_state_string, self._alphabet)
        return uncertainty, new_state_string, preds[local_idx], rank_scores[local_idx]

    def _seq_to_mut(self, seq):
        wt_seq = self._starting_sequence
        mut = ''.join(
            wt_seq[i] + str(i + 1) + seq[i] + ','
            for i in range(len(wt_seq))
            if wt_seq[i] != seq[i]
        )[:-1]
        return mut if mut else 'WT'

    def Thompson_sample(self, measured_batch):
        """GP posterior Thompson Sampling.

        Samples from the GP posterior N(mean, std) for each measured sequence
        and picks the one with the highest posterior sample. This naturally
        balances exploitation (high mean) and exploration (high uncertainty).

        """
        if len(measured_batch) == 0:
            raise RuntimeError("Cannot Thompson sample from an empty measured batch")

        sequences = [x[1] for x in measured_batch]
        encodings = self._encoder.encode(sequences)
        samples = self._model.posterior_sample(encodings, rng=self._rng)
        idx = int(np.argmax(samples))
        return sequences[idx]

    def propose_sequences(self, measured_sequences, mask=None):
        last_round = measured_sequences["round"].max()
        last_batch = measured_sequences[measured_sequences["round"] == last_round]
        _last_batch_seqs = last_batch["sequence"].tolist()
        _last_batch_true_scores = last_batch["true_score"].tolist()
        last_batch_seqs = _last_batch_seqs

        if self._recomb_rate > 0 and len(last_batch) > 1:
            last_batch_seqs = self._recombine_population(last_batch_seqs)

        measured_batch = []
        _new_seqs = []
        for seq in last_batch_seqs:
            if seq in _last_batch_seqs:
                measured_batch.append(
                    (_last_batch_true_scores[_last_batch_seqs.index(seq)], seq)
                )
            else:
                _new_seqs.append(seq)

        if len(_new_seqs) > 0:
            encodings = self._encoder.encode(_new_seqs)
            fitnesses = self._model.get_fitness(encodings)
            measured_batch.extend(
                [(fitnesses[i], _new_seqs[i]) for i in range(fitnesses.shape[0])]
            )

        measured_batch = sorted(measured_batch)
        sampled_seq = self.Thompson_sample(measured_batch)
        self._state = string_to_one_hot(sampled_seq, self._alphabet)

        initial_uncertainty = None
        samples = []
        preds = []
        rank_scores = []
        prev_cost = self._model.cost
        all_measured_seqs = set(measured_sequences["sequence"].tolist())
        no_accept_steps = 0
        max_no_accept_steps = max(20, int(self._model_queries_per_round / self._expmt_queries_per_round))

        while self._model.cost - prev_cost < self._model_queries_per_round:
            uncertainty, new_state_string, pred, rank_score = self._pick_action(
                selected_sequences=samples,
                measured_sequences=all_measured_seqs,
                mask=mask,
            )
            if new_state_string not in all_measured_seqs:
                mut = self._seq_to_mut(new_state_string)
                print(f'mut {mut} pred {pred:.4f}')
                self._best_fitness = max(self._best_fitness, pred)
                all_measured_seqs.add(new_state_string)
                samples.append(new_state_string)
                preds.append(pred)
                rank_scores.append(rank_score)
                no_accept_steps = 0
            else:
                no_accept_steps += 1
                if no_accept_steps >= max_no_accept_steps:
                    raise RuntimeError(
                        f"No new candidate accepted after {no_accept_steps} proposal steps; "
                        f"accepted={len(samples)}, measured={len(all_measured_seqs)}"
                    )
            if initial_uncertainty is None:
                initial_uncertainty = uncertainty
            if uncertainty > 2.0 * initial_uncertainty:
                sampled_seq = self.Thompson_sample(measured_batch)
                self._state = string_to_one_hot(sampled_seq, self._alphabet)
                initial_uncertainty = None

        samples = np.array(samples)
        preds = np.array(preds)
        rank_scores = np.array(rank_scores)
        samples, preds = self._select_diverse_batch(samples, preds, rank_scores)
        return measured_sequences, samples, preds

    def _log(self, measured_data, metadata, round_num, verbose, start_time):
        if self._log_file is not None:
            with open(self._log_file, "w") as f:
                json.dump(metadata, f)
                f.write("\n")
                measured_data.to_csv(f, index=False)
        if verbose:
            elapsed = time.time() - start_time
            n_measured = len(measured_data)
            top_score = measured_data["true_score"].max() if n_measured > 0 else 0
            print(f'Round {round_num}: {n_measured} measured, '
                  f'top_score={top_score:.4f}, elapsed={elapsed:.1f}s')

    def run(
        self,
        landscape,
        init_seqs=None,
        init_seqs_file=None,
        is_init_proposed_list=None,
        verbose=True,
        mask=None,
    ):
        round_num = 0
        self._model.cost = 0

        metadata = {
            "run_id": datetime.now().strftime("%H:%M:%S-%m/%d/%Y"),
            "exp_name": self._name,
            "encoder_name": self._encoder.name,
            "model_name": self._model.name,
            "landscape_name": landscape.name,
            "rounds": self._rounds,
            "expmt_queries_per_round": self._expmt_queries_per_round,
            "model_queries_per_round": self._model_queries_per_round,
        }

        if init_seqs is not None or init_seqs_file is not None:
            if init_seqs_file is not None:
                assert init_seqs_file.endswith(".csv")
                df = pd.read_csv(init_seqs_file)
                init_seqs = df["Variants"].to_numpy()
                if "is_init_proposed_seq" in df.columns:
                    is_init_proposed_list = df["is_init_proposed_seq"].tolist()
            true_score = landscape.get_fitness(init_seqs)
            df = pd.DataFrame({
                "sequence": init_seqs,
                "true_score": true_score,
                "is_init_proposed": is_init_proposed_list if is_init_proposed_list is not None else [True] * len(init_seqs),
            })
            df.dropna(inplace=True)
            true_score = df["true_score"].to_numpy()
            seqs = df["sequence"].values
            props = df["is_init_proposed"].values
            measured_data = pd.DataFrame({
                "sequence": seqs,
                "model_score": np.nan,
                "true_score": true_score,
                "round": round_num,
                "model_cost": 0,
                "measurement_cost": len(seqs),
                "is_init_proposed": props,
            })
            self._log(measured_data, metadata, round_num, verbose, time.time())
            measured_data = measured_data.drop_duplicates(subset=['sequence'], keep='last')
            measured_data = measured_data.dropna(subset=['true_score'])
            measured_data = measured_data.reset_index(drop=True)
        else:
            measured_data = pd.DataFrame({
                "sequence": [self._starting_sequence],
                "model_score": [np.nan],
                "true_score": [1],
                "round": round_num,
                "model_cost": [0],
                "measurement_cost": [1],
                "is_init_proposed": [True],
            })
            self._log(measured_data, metadata, round_num, verbose, time.time())

        # Train on all available data
        training_data = measured_data[measured_data["round"] <= round_num]
        if 'true_score' in training_data.columns:
            training_data = training_data.sort_values(by="true_score", ascending=False)
        encodings = self._encoder.encode(training_data["sequence"].to_list())
        labels = training_data["true_score"].to_numpy()
        if len(labels) > 0:
            self._model.train(encodings, labels, verbose=verbose)

        if verbose:
            print(f'Round {round_num} proposing...')

        proposal_seed_data = measured_data
        if len(proposal_seed_data) > 0:
            best_idx = proposal_seed_data["true_score"].idxmax()
            print(f'Starting from: {proposal_seed_data.loc[best_idx, "sequence"]}')

        measured_data, seqs, preds = self.propose_sequences(proposal_seed_data, mask=mask)
        if verbose:
            print(f'Round {round_num} measuring...')
        true_score = landscape.get_fitness(seqs)

        df = pd.DataFrame({
            "sequence": seqs,
            "true_score": true_score,
            "preds": preds,
        })
        df.dropna(inplace=True)
        true_score = df["true_score"].to_numpy()
        preds = df["preds"].to_numpy()
        seqs = df["sequence"].values

        if len(seqs) > 0:
            new_data = pd.DataFrame({
                "sequence": seqs,
                "model_score": preds,
                "true_score": true_score,
                "round": self._rounds,
                "model_cost": self._model.cost,
                "measurement_cost": len(measured_data) + len(seqs),
                "is_init_proposed": False,
            })
            for record in new_data.to_dict('records'):
                measured_data.loc[len(measured_data)] = record
        self._log(measured_data, metadata, self._rounds, verbose, time.time())

        return measured_data, metadata


class MCMC:
    """Markov Chain Monte Carlo explorer.

    Algorithm:
        1. Thompson sample starting sequence
        2. Initialize batch of states with random mutations
        3. Iteratively mutate states and accept/reject via Metropolis-Hastings
        4. Return top expmt_queries_per_round sequences
    """

    def __init__(
        self,
        encoder,
        model,
        rounds: int,
        expmt_queries_per_round: int,
        model_queries_per_round: int,
        starting_sequence: str,
        alphabet: str = AAS,
        log_file: Optional[str] = None,
        seed: int = 0,
        mu: float = 1.2,
        temperature: float = 0.06,
        batch_size: int = 10,
    ):
        self._name = f"MCSA={mu}"
        self._encoder = encoder
        self._model = model
        self._rounds = rounds
        self._expmt_queries_per_round = expmt_queries_per_round
        self._model_queries_per_round = model_queries_per_round
        self._starting_sequence = starting_sequence
        self._log_file = log_file
        self._rng = np.random.default_rng(seed)
        self._alphabet = alphabet
        self._state = string_to_one_hot(starting_sequence, alphabet)
        self._seq_len = len(starting_sequence)
        self._mu = mu
        self._temperature = temperature
        self._batch_size = batch_size

    def init_states(self, state, mask=None):
        actions = set()
        pos_changes = []
        for pos in range(self._seq_len):
            pos_changes.append([])
            for res in range(len(self._alphabet)):
                if mask is None or not mask[res, pos]:
                    pos_changes[pos].append((pos, res))

        pos_changes_len = len([x for x in pos_changes if len(x) > 0])
        pos_changes_list = [pos for pos, x in enumerate(pos_changes) if len(x) > 0]
        max_tries = 5000
        tries = 0
        while len(actions) < self._batch_size:
            action = []
            m = self._rng.poisson(1)
            m = max(1, min(m, pos_changes_len))
            for pos in self._rng.choice(pos_changes_list, m, replace=False):
                pos_tuple = tuple(self._rng.choice(pos_changes[pos]))
                action.append(pos_tuple)
            if len(action) > 0 and tuple(action) not in actions:
                actions.add(tuple(action))
            tries += 1
            if tries > max_tries:
                break

        if len(actions) < self._batch_size:
            self._batch_size = len(actions)

        actions = list(actions)
        states = []
        states_str = []
        for i in range(self._batch_size):
            x = np.zeros((self._seq_len, len(self._alphabet)))
            for action in actions[i]:
                x[action] = 1
            state = construct_mutant_from_sample(x, state)
            states.append(state)
            states_str.append(one_hot_to_string(state, self._alphabet))
        fitness, uncertainty = self._fitness(states)
        return states, states_str, fitness, uncertainty

    def mutate(self, states, mask=None):
        new_states = []
        for state in states:
            pos_changes = []
            for pos in range(self._seq_len):
                pos_changes.append([])
                for res in range(len(self._alphabet)):
                    if mask is None or not mask[res, pos]:
                        pos_changes[pos].append((pos, res))
            pos_changes_len = len([x for x in pos_changes if len(x) > 0])
            pos_changes_list = [pos for pos, x in enumerate(pos_changes) if len(x) > 0]

            action = []
            m = self._rng.poisson(self._rng.uniform(1, self._mu))
            m = max(1, min(m, pos_changes_len))
            for pos in self._rng.choice(pos_changes_list, m, replace=False):
                pos_tuple = tuple(self._rng.choice(pos_changes[pos]))
                action.append(pos_tuple)
            x = np.zeros((self._seq_len, len(self._alphabet)))
            for act in action:
                x[act] = 1
            new_states.append(construct_mutant_from_sample(x, state))
        return new_states

    def _fitness(self, states):
        states_str = [one_hot_to_string(state, self._alphabet) for state in states]
        states_encoding = self._encoder.encode(states_str)
        preds = self._model.get_fitness(states_encoding)
        uncertainties = self._model.uncertainties
        return preds, uncertainties

    def update_states(self, states, fitness, new_states, init_uncertainty):
        new_fitness, new_uncertainty = self._fitness(new_states)
        y_star = np.where(new_uncertainty > 2.0 * init_uncertainty, -np.inf, new_fitness)
        prob_margin = np.minimum(1.0, np.exp((y_star - fitness) / self._temperature))
        for i in range(self._batch_size):
            if self._rng.random() < prob_margin[i]:
                states[i] = new_states[i]
                fitness[i] = new_fitness[i]
        seqs = [one_hot_to_string(state, self._alphabet) for state in states]
        return states, seqs, fitness

    def _seq_to_mut(self, seq):
        wt_seq = self._starting_sequence
        mut = ''.join(
            wt_seq[i] + str(i + 1) + seq[i] + ','
            for i in range(len(wt_seq))
            if wt_seq[i] != seq[i]
        )[:-1]
        return mut if mut else 'WT'

    def Thompson_sample(self, measured_batch):
        """GP posterior Thompson Sampling.

        Samples from the GP posterior N(mean, std) for each measured sequence
        and picks the one with the highest posterior sample.
        """
        sequences = measured_batch["sequence"].tolist()
        if len(sequences) == 0:
            raise RuntimeError("Cannot Thompson sample from an empty measured batch")

        encodings = self._encoder.encode(sequences)
        samples = self._model.posterior_sample(encodings, rng=self._rng)
        idx = int(np.argmax(samples))
        return sequences[idx]

    def propose_sequences(self, measured_sequences, mask=None):
        last_batch = measured_sequences
        measured_batch = last_batch[["true_score", "sequence"]].copy()
        measured_batch = measured_batch.sort_values(by="true_score", ascending=False)
        measured_batch = measured_batch.reset_index(drop=True)

        sampled_seq = self.Thompson_sample(measured_batch)
        self._state = string_to_one_hot(sampled_seq, self._alphabet)

        samples = []
        preds = []
        prev_cost = self._model.cost
        all_measured_seqs = set(measured_sequences["sequence"].tolist())

        states, seqs, fitness, init_uncertainty = self.init_states(self._state, mask=mask)
        print('### init_states ###')
        for seq, pred in zip(seqs, fitness):
            if seq not in all_measured_seqs:
                mut = self._seq_to_mut(seq)
                print(f'mut {mut} pred {pred:.4f}')
                all_measured_seqs.add(seq)
                samples.append(seq)
                preds.append(pred)

        print('### mutate ###')
        while self._model.cost - prev_cost < self._model_queries_per_round:
            new_states = self.mutate(states, mask=mask)
            states, seqs, fitness = self.update_states(states, fitness, new_states, init_uncertainty)
            for seq, pred in zip(seqs, fitness):
                if seq not in all_measured_seqs:
                    mut = self._seq_to_mut(seq)
                    print(f'mut {mut} pred {pred:.4f}')
                    all_measured_seqs.add(seq)
                    samples.append(seq)
                    preds.append(pred)

        samples = np.array(samples)
        preds = np.array(preds)
        sorted_order = np.argsort(preds)[:-self._expmt_queries_per_round - 1:-1]
        return measured_sequences, samples[sorted_order], preds[sorted_order]

    def _log(self, measured_data, metadata, round_num, verbose, start_time):
        if self._log_file is not None:
            with open(self._log_file, "w") as f:
                json.dump(metadata, f)
                f.write("\n")
                measured_data.to_csv(f, index=False)
        if verbose:
            elapsed = time.time() - start_time
            n_measured = len(measured_data)
            top_score = measured_data["true_score"].max() if n_measured > 0 else 0
            print(f'Round {round_num}: {n_measured} measured, '
                  f'top_score={top_score:.4f}, elapsed={elapsed:.1f}s')

    def run(
        self,
        landscape,
        init_seqs=None,
        init_seqs_file=None,
        is_init_proposed_list=None,
        verbose=True,
        mask=None,
    ):
        round_num = 0
        self._model.cost = 0

        metadata = {
            "run_id": datetime.now().strftime("%H:%M:%S-%m/%d/%Y"),
            "exp_name": self._name,
            "encoder_name": self._encoder.name,
            "model_name": self._model.name,
            "landscape_name": landscape.name,
            "rounds": self._rounds,
            "expmt_queries_per_round": self._expmt_queries_per_round,
            "model_queries_per_round": self._model_queries_per_round,
        }

        if init_seqs is not None or init_seqs_file is not None:
            if init_seqs_file is not None:
                assert init_seqs_file.endswith(".csv")
                df = pd.read_csv(init_seqs_file)
                init_seqs = df["Variants"].to_numpy()
                if "is_init_proposed_seq" in df.columns:
                    is_init_proposed_list = df["is_init_proposed_seq"].tolist()
            true_score = landscape.get_fitness(init_seqs)
            df = pd.DataFrame({
                "sequence": init_seqs,
                "true_score": true_score,
                "is_init_proposed": is_init_proposed_list if is_init_proposed_list is not None else [True] * len(init_seqs),
            })
            df.dropna(inplace=True)
            true_score = df["true_score"].to_numpy()
            seqs = df["sequence"].values
            props = df["is_init_proposed"].values
            measured_data = pd.DataFrame({
                "sequence": seqs,
                "model_score": np.nan,
                "true_score": true_score,
                "round": round_num,
                "model_cost": 0,
                "measurement_cost": len(seqs),
                "is_init_proposed": props,
            })
            self._log(measured_data, metadata, round_num, verbose, time.time())
            measured_data = measured_data.drop_duplicates(subset=['sequence'], keep='last')
            measured_data = measured_data.dropna(subset=['true_score'])
            measured_data = measured_data.reset_index(drop=True)
        else:
            measured_data = pd.DataFrame({
                "sequence": [self._starting_sequence],
                "model_score": [np.nan],
                "true_score": [1],
                "round": round_num,
                "model_cost": [0],
                "measurement_cost": [1],
                "is_init_proposed": [True],
            })
            self._log(measured_data, metadata, round_num, verbose, time.time())

        # Train on all available data
        training_data = measured_data[measured_data["round"] <= round_num]
        encodings = self._encoder.encode(training_data["sequence"].to_list())
        labels = training_data["true_score"].to_numpy()
        if len(labels) > 0:
            self._model.train(encodings, labels, verbose=verbose)

        if verbose:
            print(f'Round {round_num} proposing...')

        proposal_seed_data = measured_data
        if len(proposal_seed_data) > 0:
            best_idx = proposal_seed_data["true_score"].idxmax()
            print(f'Starting from: {proposal_seed_data.loc[best_idx, "sequence"]}')

        measured_data, seqs, preds = self.propose_sequences(proposal_seed_data, mask=mask)
        if len(seqs) > self._expmt_queries_per_round:
            warnings.warn("Proposed more sequences than expmt_queries_per_round")

        if verbose:
            print(f'Round {round_num} measuring...')
        true_score = landscape.get_fitness(seqs)

        df = pd.DataFrame({
            "sequence": seqs,
            "true_score": true_score,
            "preds": preds,
        })
        df.dropna(inplace=True)
        true_score = df["true_score"].to_numpy()
        preds = df["preds"].to_numpy()
        seqs = df["sequence"].values

        if len(seqs) > 0:
            new_data = pd.DataFrame({
                "sequence": seqs,
                "model_score": preds,
                "true_score": true_score,
                "round": self._rounds,
                "model_cost": self._model.cost,
                "measurement_cost": len(measured_data) + len(seqs),
                "is_init_proposed": False,
            })
            for record in new_data.to_dict('records'):
                measured_data.loc[len(measured_data)] = record
        self._log(measured_data, metadata, self._rounds, verbose, time.time())

        return measured_data, metadata


class Boltz2BO:
    """Pool-based BO explorer for Boltz2 384-dim pooled embeddings.

    Instead of iterative single-action selection (BO_EVO), generates a large
    candidate pool upfront, batch-encodes all candidates with one Boltz2 API
    call, then screens with the GP surrogate.

    Key differences from BO_EVO:
    - Pool-based batch screening (1-2 API calls vs hundreds)
    - No ARD position sampling (pooled embedding is position-agnostic)
    - Optional embedding cosine distance for batch diversity
    - Pre-filters candidates by synthesis penalty before encoding
    """

    def __init__(
        self,
        encoder,
        model,
        rounds: int,
        expmt_queries_per_round: int,
        model_queries_per_round: int,
        starting_sequence: str,
        alphabet: str = AAS,
        log_file: Optional[str] = None,
        seed: int = 0,
        util_func: str = "UCB",
        uf_param: float = 0.2,
        min_mutations: int = 1,
        max_mutations: Optional[int] = None,
        batch_diversity_min_hamming: int = 2,
        proposal_synthesis_penalty_weight: float = 0.25,
        proposal_max_synthesis_penalty: float = 3.1,
        diversity_mode: str = "auto",
        min_cosine_distance: float = 0.03,
        auxiliary_encoder=None,
        auxiliary_model=None,
    ):
        if expmt_queries_per_round < 1:
            raise ValueError("expmt_queries_per_round must be positive")
        if model_queries_per_round < expmt_queries_per_round:
            raise ValueError("model_queries_per_round must be >= expmt_queries_per_round")

        self._name = f"Boltz2BO_acq={util_func}_diversity={diversity_mode}"
        self._encoder = encoder
        self._model = model
        self._rounds = rounds
        self._expmt_queries_per_round = expmt_queries_per_round
        self._model_queries_per_round = model_queries_per_round
        self._starting_sequence = starting_sequence
        self._log_file = log_file
        self._rng = np.random.default_rng(seed)
        self._alphabet = alphabet
        self._min_mutations = max(1, int(min_mutations))
        self._max_mutations = int(max_mutations) if max_mutations else min(3, len(starting_sequence))
        self._batch_diversity_min_hamming = int(batch_diversity_min_hamming)
        self._proposal_synthesis_penalty_weight = float(proposal_synthesis_penalty_weight)
        self._proposal_max_synthesis_penalty = float(proposal_max_synthesis_penalty)
        self._diversity_mode = diversity_mode
        self._min_cosine_distance = float(min_cosine_distance)
        self._best_fitness = 0.0
        self._seq_len = len(starting_sequence)
        self._aux_encoder = auxiliary_encoder
        self._aux_model = auxiliary_model
        self._pos_weights = np.full(self._seq_len, 1.0 / self._seq_len)

        util_funcs = {
            "UCB": acq.UCB, "LCB": acq.LCB, "EI": acq.EI,
            "PI": acq.PI, "TS": acq.TS, "Greedy": acq.Greedy,
            "NEI": acq.NEI, "QUCB": acq.QUCB,
        }
        if util_func not in util_funcs:
            raise ValueError(f"Unsupported acquisition function: {util_func}")
        self._util_func = util_funcs[util_func]
        self._uf_param = uf_param

        # Validate mutation space
        aa_options = len(alphabet) - 1
        capacity = 0
        for k in range(self._min_mutations, self._max_mutations + 1):
            if k > self._seq_len:
                break
            capacity += math.comb(self._seq_len, k) * (aa_options ** k)
        if capacity < expmt_queries_per_round:
            raise ValueError(
                f"Mutation space too small ({capacity} candidates); "
                f"need >= {expmt_queries_per_round}"
            )

    def _update_pos_weights(self):
        """Update per-position mutation weights from auxiliary GP's ARD."""
        if self._aux_encoder is None or self._aux_model is None:
            return
        if not hasattr(self._aux_model, "ard_lengthscale"):
            return
        ls = np.asarray(self._aux_model.ard_lengthscale, dtype=float)
        per_position_dim = getattr(self._aux_encoder, "per_position_dim", None)
        if per_position_dim is None or len(ls) != self._seq_len * per_position_dim:
            return
        ls_per_pos = ls.reshape(self._seq_len, per_position_dim)
        relevance = 1.0 / (ls_per_pos.mean(axis=1) + 1e-8)
        total = relevance.sum()
        if not np.isfinite(total) or total <= 0:
            return
        self._pos_weights = relevance / total

    def _generate_pool(self, seed_sequences, measured_seqs, pool_size):
        """Generate a diverse candidate pool from seed sequences."""
        self._update_pos_weights()
        pool = []
        seen = set()
        max_attempts = pool_size * 10
        attempts = 0

        while len(pool) < pool_size and attempts < max_attempts:
            seed = seed_sequences[self._rng.integers(len(seed_sequences))]
            seq = list(seed)
            n_mut = max(self._min_mutations, self._rng.poisson(1))
            n_mut = min(n_mut, self._max_mutations, self._seq_len)
            positions = self._rng.choice(
                self._seq_len, size=n_mut, replace=False,
                p=self._pos_weights,
            )
            for pos in positions:
                choices = [aa for aa in self._alphabet if aa != seq[pos]]
                seq[pos] = choices[int(self._rng.integers(len(choices)))]
            mutant = ''.join(seq)
            if mutant in seen or mutant in measured_seqs:
                attempts += 1
                continue
            penalty = float(synthesis_metrics(mutant)['synthesis_penalty'])
            if penalty > self._proposal_max_synthesis_penalty:
                attempts += 1
                continue
            seen.add(mutant)
            pool.append(mutant)
            attempts += 1

        return pool

    def _select_diverse_batch(self, pool, encodings, rank_scores):
        """Select top-k diverse candidates using Hamming or embedding cosine distance."""
        if len(pool) == 0:
            raise RuntimeError("No candidate sequences in pool")

        finite = np.where(np.isfinite(rank_scores))[0]
        if len(finite) == 0:
            raise RuntimeError("No finite-ranked candidates")

        jitter = self._rng.normal(0.0, 1e-9, size=len(rank_scores))
        order = np.argsort(rank_scores + jitter)[::-1]

        use_cosine = self._diversity_mode == "cosine" or (
            self._diversity_mode == "auto"
            and encodings is not None
            and len(encodings.shape) == 2
            and encodings.shape[1] > 20
        )

        selected = []
        selected_normed = []
        selected_rank = []

        for idx in order:
            if not np.isfinite(rank_scores[idx]):
                continue
            seq = pool[idx]

            if use_cosine and len(selected) > 0:
                emb = encodings[idx]
                emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
                too_close = False
                for sel_n in selected_normed:
                    if np.dot(emb_norm, sel_n) > (1.0 - self._min_cosine_distance):
                        too_close = True
                        break
                if too_close:
                    continue

            elif self._batch_diversity_min_hamming > 0 and len(selected) > 0:
                if any(
                    _hamming_distance(seq, kept) < self._batch_diversity_min_hamming
                    for kept in selected
                ):
                    continue

            selected.append(seq)
            if use_cosine:
                emb = encodings[idx]
                selected_normed.append(emb / (np.linalg.norm(emb) + 1e-10))
            selected_rank.append(rank_scores[idx])

            if len(selected) >= self._expmt_queries_per_round:
                break

        # Cosine too strict -> fall back to Hamming
        if len(selected) < self._expmt_queries_per_round and use_cosine:
            print(f"  Cosine diversity yielded only {len(selected)} candidates, "
                  "falling back to Hamming")
            selected = []
            selected_rank = []
            for idx in order:
                if not np.isfinite(rank_scores[idx]):
                    continue
                seq = pool[idx]
                if self._batch_diversity_min_hamming > 0 and len(selected) > 0:
                    if any(
                        _hamming_distance(seq, kept) < self._batch_diversity_min_hamming
                        for kept in selected
                    ):
                        continue
                selected.append(seq)
                selected_rank.append(rank_scores[idx])
                if len(selected) >= self._expmt_queries_per_round:
                    break

        if len(selected) < self._expmt_queries_per_round:
            raise RuntimeError(
                f"Only {len(selected)} diverse candidates found; "
                f"need {self._expmt_queries_per_round}"
            )

        return np.array(selected), np.array(selected_rank)

    def Thompson_sample(self, measured_batch):
        """GP posterior Thompson Sampling."""
        sequences = [x[1] for x in measured_batch]
        encodings = self._encoder.encode(sequences)
        samples = self._model.posterior_sample(encodings, rng=self._rng)
        idx = int(np.argmax(samples))
        return sequences[idx]

    def propose_sequences(self, measured_sequences, mask=None):
        last_round = measured_sequences["round"].max()
        last_batch = measured_sequences[measured_sequences["round"] == last_round]
        last_batch_seqs = last_batch["sequence"].tolist()
        last_batch_scores = last_batch["true_score"].tolist()

        measured_batch = list(zip(last_batch_scores, last_batch_seqs))
        measured_batch.sort(key=lambda x: x[0])

        sampled_seq = self.Thompson_sample(measured_batch)
        sorted_batch = sorted(measured_batch, key=lambda x: x[0], reverse=True)
        seeds = [sampled_seq] + [s for _, s in sorted_batch[:5]]
        seeds = list(dict.fromkeys(seeds))

        all_measured = set(measured_sequences["sequence"].tolist())
        pool_size = self._model_queries_per_round

        print(f"  Generating candidate pool (target {pool_size})...")
        pool = self._generate_pool(seeds, all_measured, pool_size)
        print(f"  Pool generated: {len(pool)} feasible candidates")

        if len(pool) < self._expmt_queries_per_round:
            raise RuntimeError(
                f"Only {len(pool)} feasible candidates generated; "
                f"need {self._expmt_queries_per_round}"
            )

        # Batch encode all candidates (1-2 API calls)
        print(f"  Encoding {len(pool)} candidates with {self._encoder.name}...")
        encodings = self._encoder.encode(pool)

        # GP predict all at once
        preds = self._model.get_fitness(encodings)
        uncertainties = self._model.uncertainties

        # Acquisition function
        kwargs = {"best_val": self._best_fitness, "rng": self._rng}
        utility = self._util_func(
            preds, uncertainties, h_param=self._uf_param, **kwargs
        )

        # Synthesis penalty adjustment
        penalties = np.array([
            float(synthesis_metrics(seq)['synthesis_penalty']) for seq in pool
        ])
        rank_scores = utility - self._proposal_synthesis_penalty_weight * penalties

        # Diversity selection
        selected_seqs, _ = self._select_diverse_batch(pool, encodings, rank_scores)

        # Map selected back to GP predictions
        seq_to_pred = dict(zip(pool, preds))
        selected_preds = np.array([seq_to_pred[seq] for seq in selected_seqs])

        if len(selected_preds) > 0:
            self._best_fitness = max(self._best_fitness, float(np.max(selected_preds)))

        return measured_sequences, selected_seqs, selected_preds

    def _seq_to_mut(self, seq):
        wt_seq = self._starting_sequence
        mut = ''.join(
            wt_seq[i] + str(i + 1) + seq[i] + ','
            for i in range(len(wt_seq))
            if wt_seq[i] != seq[i]
        )[:-1]
        return mut if mut else 'WT'

    def _log(self, measured_data, metadata, round_num, verbose, start_time):
        if self._log_file is not None:
            with open(self._log_file, "w") as f:
                json.dump(metadata, f)
                f.write("\n")
                measured_data.to_csv(f, index=False)
        if verbose:
            elapsed = time.time() - start_time
            n_measured = len(measured_data)
            top_score = measured_data["true_score"].max() if n_measured > 0 else 0
            print(f'Round {round_num}: {n_measured} measured, '
                  f'top_score={top_score:.4f}, elapsed={elapsed:.1f}s')

    def run(
        self,
        landscape,
        init_seqs=None,
        init_seqs_file=None,
        is_init_proposed_list=None,
        verbose=True,
        mask=None,
    ):
        round_num = 0
        self._model.cost = 0

        metadata = {
            "run_id": datetime.now().strftime("%H:%M:%S-%m/%d/%Y"),
            "exp_name": self._name,
            "encoder_name": self._encoder.name,
            "model_name": self._model.name,
            "landscape_name": landscape.name,
            "rounds": self._rounds,
            "expmt_queries_per_round": self._expmt_queries_per_round,
            "model_queries_per_round": self._model_queries_per_round,
        }

        if init_seqs is not None or init_seqs_file is not None:
            if init_seqs_file is not None:
                assert init_seqs_file.endswith(".csv")
                df = pd.read_csv(init_seqs_file)
                init_seqs = df["Variants"].to_numpy()
                if "is_init_proposed_seq" in df.columns:
                    is_init_proposed_list = df["is_init_proposed_seq"].tolist()
            true_score = landscape.get_fitness(init_seqs)
            df = pd.DataFrame({
                "sequence": init_seqs,
                "true_score": true_score,
                "is_init_proposed": is_init_proposed_list if is_init_proposed_list is not None else [True] * len(init_seqs),
            })
            df.dropna(inplace=True)
            true_score = df["true_score"].to_numpy()
            seqs = df["sequence"].values
            props = df["is_init_proposed"].values
            measured_data = pd.DataFrame({
                "sequence": seqs,
                "model_score": np.nan,
                "true_score": true_score,
                "round": round_num,
                "model_cost": 0,
                "measurement_cost": len(seqs),
                "is_init_proposed": props,
            })
            self._log(measured_data, metadata, round_num, verbose, time.time())
            measured_data = measured_data.drop_duplicates(subset=['sequence'], keep='last')
            measured_data = measured_data.dropna(subset=['true_score'])
            measured_data = measured_data.reset_index(drop=True)
        else:
            measured_data = pd.DataFrame({
                "sequence": [self._starting_sequence],
                "model_score": [np.nan],
                "true_score": [1],
                "round": round_num,
                "model_cost": [0],
                "measurement_cost": [1],
                "is_init_proposed": [True],
            })
            self._log(measured_data, metadata, round_num, verbose, time.time())

        training_data = measured_data[measured_data["round"] <= round_num]
        if 'true_score' in training_data.columns:
            training_data = training_data.sort_values(by="true_score", ascending=False)
        encodings = self._encoder.encode(training_data["sequence"].to_list())
        labels = training_data["true_score"].to_numpy()
        if len(labels) > 0:
            self._model.train(encodings, labels, verbose=verbose)
        if self._aux_encoder is not None and self._aux_model is not None and len(labels) > 0:
            aux_enc = self._aux_encoder.encode(training_data["sequence"].to_list())
            self._aux_model.train(aux_enc, labels, verbose=False)

        if verbose:
            print(f'Round {round_num} proposing...')

        proposal_seed_data = measured_data
        if len(proposal_seed_data) > 0:
            best_idx = proposal_seed_data["true_score"].idxmax()
            print(f'Starting from: {proposal_seed_data.loc[best_idx, "sequence"]}')

        measured_data, seqs, preds = self.propose_sequences(proposal_seed_data, mask=mask)
        if verbose:
            print(f'Round {round_num} measuring...')
        true_score = landscape.get_fitness(seqs)

        df = pd.DataFrame({
            "sequence": seqs,
            "true_score": true_score,
            "preds": preds,
        })
        df.dropna(inplace=True)
        true_score = df["true_score"].to_numpy()
        preds = df["preds"].to_numpy()
        seqs = df["sequence"].values

        if len(seqs) > 0:
            new_data = pd.DataFrame({
                "sequence": seqs,
                "model_score": preds,
                "true_score": true_score,
                "round": self._rounds,
                "model_cost": self._model.cost,
                "measurement_cost": len(measured_data) + len(seqs),
                "is_init_proposed": False,
            })
            for record in new_data.to_dict('records'):
                measured_data.loc[len(measured_data)] = record
        self._log(measured_data, metadata, self._rounds, verbose, time.time())

        return measured_data, metadata

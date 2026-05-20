"""Gaussian Process surrogate model for Bayesian Optimization."""
import math

import numpy as np
import torch


class GPRegressor:
    """Gaussian Process regressor with ARD RBF/Matern kernels.

    Optimizes log length scales, signal variance and noise by maximizing the
    log marginal likelihood for the current BO observations.
    """

    def __init__(self, kernel="Matern", training_iters=120, lr=0.05, device=None):
        if kernel not in {"Matern", "RBF"}:
            raise ValueError(f"Unsupported GP kernel: {kernel}")
        self.name = "GPRegressor"
        self._kernel_name = kernel
        self._training_iters = int(training_iters)
        self._lr = float(lr)
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._dtype = torch.float32
        self._uncertainties = np.inf
        self.cost = 0

        self._X = None
        self._y = None
        self._y_mean = None
        self._y_std = None
        self._log_lengthscale = None
        self._log_signal = None
        self._log_noise = None
        self._alpha = None
        self._chol = None
        self.log_marginal_likelihood_value_ = None

    def _as_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(device=self._device, dtype=self._dtype)
        return torch.as_tensor(x, device=self._device, dtype=self._dtype)

    def _scaled_distance(self, xa, xb):
        lengthscale = torch.exp(self._log_lengthscale).clamp_min(1e-4)
        xa = xa / lengthscale
        xb = xb / lengthscale
        diff = xa[:, None, :] - xb[None, :, :]
        return torch.sqrt(torch.sum(diff * diff, dim=-1).clamp_min(1e-12))

    def _kernel(self, xa, xb):
        r = self._scaled_distance(xa, xb)
        signal2 = torch.exp(2.0 * self._log_signal).clamp_min(1e-8)
        if self._kernel_name == "RBF":
            return signal2 * torch.exp(-0.5 * r * r)
        sqrt5 = math.sqrt(5.0)
        return signal2 * (1.0 + sqrt5 * r + 5.0 * r * r / 3.0) * torch.exp(-sqrt5 * r)

    def _safe_cholesky(self, k):
        for jitter in [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            try:
                return torch.linalg.cholesky(k + jitter * torch.eye(k.shape[0], device=k.device, dtype=k.dtype))
            except torch._C._LinAlgError:
                continue
        raise torch._C._LinAlgError("Cholesky failed even with jitter=1e-2")

    def _negative_log_marginal_likelihood(self):
        n = self._X.shape[0]
        eye = torch.eye(n, device=self._device, dtype=self._dtype)
        noise2 = torch.exp(2.0 * self._log_noise).clamp_min(1e-8)
        k = self._kernel(self._X, self._X) + (noise2 + 1e-6) * eye
        chol = self._safe_cholesky(k)
        alpha = torch.cholesky_solve(self._y[:, None], chol).squeeze(-1)
        data_fit = 0.5 * torch.dot(self._y, alpha)
        complexity = torch.sum(torch.log(torch.diagonal(chol)))
        constant = 0.5 * n * math.log(2.0 * math.pi)
        return data_fit + complexity + constant

    def train(self, X, y, verbose=True):
        if np.asarray(X).ndim == 1:
            X = np.asarray(X).reshape(1, -1)
        X_t = self._as_tensor(X)
        y_t = self._as_tensor(y).flatten()
        if X_t.shape[0] != y_t.shape[0]:
            raise ValueError(f"X/y size mismatch: {X_t.shape[0]} vs {y_t.shape[0]}")

        self._X = X_t
        self._y_mean = y_t.mean()
        y_std = y_t.std(unbiased=False)
        self._y_std = torch.clamp(y_std, min=torch.tensor(1.0, device=self._device))
        self._y = (y_t - self._y_mean) / self._y_std

        d = X_t.shape[1]
        if self._log_lengthscale is None or self._log_lengthscale.numel() != d:
            self._log_lengthscale = torch.nn.Parameter(torch.zeros(d, device=self._device, dtype=self._dtype))
            self._log_signal = torch.nn.Parameter(torch.tensor(0.0, device=self._device, dtype=self._dtype))
            self._log_noise = torch.nn.Parameter(torch.tensor(-4.0, device=self._device, dtype=self._dtype))

        params = [self._log_lengthscale, self._log_signal, self._log_noise]
        opt = torch.optim.Adam(params, lr=self._lr)
        for _ in range(self._training_iters):
            opt.zero_grad(set_to_none=True)
            loss = self._negative_log_marginal_likelihood()
            loss.backward()
            opt.step()
            with torch.no_grad():
                self._log_lengthscale.clamp_(math.log(1e-3), math.log(1e5))
                self._log_signal.clamp_(math.log(1e-3), math.log(1e3))
                self._log_noise.clamp_(math.log(1e-5), math.log(1.0))

        with torch.no_grad():
            n = self._X.shape[0]
            eye = torch.eye(n, device=self._device, dtype=self._dtype)
            noise2 = torch.exp(2.0 * self._log_noise).clamp_min(1e-8)
            k = self._kernel(self._X, self._X) + (noise2 + 1e-6) * eye
            self._chol = self._safe_cholesky(k)
            self._alpha = torch.cholesky_solve(self._y[:, None], self._chol).squeeze(-1)
            nll = self._negative_log_marginal_likelihood()
            self.log_marginal_likelihood_value_ = float((-nll).detach().cpu())
        if verbose:
            print(
                f"GP trained on {len(y_t)} samples, "
                f"log-marginal-likelihood: {self.log_marginal_likelihood_value_:.3f}"
            )

    def predict(self, X, return_std=False):
        if self._X is None or self._chol is None or self._alpha is None:
            raise RuntimeError("GPRegressor.predict called before train")
        if np.asarray(X).ndim == 1:
            X = np.asarray(X).reshape(1, -1)
        X_t = self._as_tensor(X)
        with torch.no_grad():
            k_star = self._kernel(X_t, self._X)
            mean = k_star @ self._alpha
            v = torch.cholesky_solve(k_star.T, self._chol)
            k_xx = torch.diagonal(self._kernel(X_t, X_t))
            var = (k_xx - torch.sum(k_star * v.T, dim=1)).clamp_min(1e-9)
            mean = mean * self._y_std + self._y_mean
            std = torch.sqrt(var) * self._y_std
        mean_np = mean.detach().cpu().numpy()
        std_np = std.detach().cpu().numpy()
        if return_std:
            return mean_np, std_np
        return mean_np

    def get_fitness(self, X):
        mean, std = self.predict(X, return_std=True)
        self._uncertainties = std
        self.cost += len(mean)
        return mean

    def posterior_sample(self, X, rng=None):
        mean, std = self.predict(X, return_std=True)
        std = np.maximum(std, 1e-8)
        rng = rng or np.random.default_rng()
        return rng.normal(mean, std)

    @property
    def uncertainties(self):
        return self._uncertainties

    @property
    def ard_lengthscale(self):
        if self._log_lengthscale is None:
            raise RuntimeError("ARD length scales are unavailable before training")
        return torch.exp(self._log_lengthscale).detach().cpu().numpy()

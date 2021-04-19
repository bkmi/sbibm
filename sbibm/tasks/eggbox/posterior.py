import math
from functools import cached_property, lru_cache, partial
from numbers import Number
from typing import Callable

import numpy as np
import scipy
import scipy.stats
import torch
from joblib import Parallel, delayed, parallel_backend
from toolz import compose, identity
from torch.distributions import Normal, Uniform, constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


def g(parameters):
    if isinstance(parameters, torch.Tensor):
        return torch.sin(math.pi * parameters).pow(2)
    else:
        return np.sin(np.pi * parameters) ** 2


@lru_cache
def get_indicator_function(lower_x: float, upper_x: float, fn: Callable = identity):
    return lambda x: np.where((lower_x <= x) & (x <= upper_x), fn(x), np.zeros_like(x))


@lru_cache
def get_unnormalized_posterior_pdf(x: float, sigma: float = 0.1) -> np.ndarray:
    normal_pdf = compose(np.exp, scipy.stats.norm(loc=x, scale=sigma).logpdf)
    return get_indicator_function(
        0.0, 1.0, lambda theta: normal_pdf(np.sin(np.pi * theta) ** 2)
    )


@lru_cache
def get_posterior_pdf(x: float, sigma: float = 0.1) -> np.ndarray:
    unnormalized_posterior_pdf = get_unnormalized_posterior_pdf(x, sigma)
    evidence = get_evidence(x, sigma)

    def posterior_pdf(value):
        return unnormalized_posterior_pdf(value) / evidence

    return posterior_pdf


@lru_cache
def get_evidence(x: float, sigma: float = 0.1) -> np.ndarray:
    posterior_pdf = get_unnormalized_posterior_pdf(x, sigma)
    return scipy.integrate.quad(posterior_pdf, 0, 1)[0]


@lru_cache
def get_posterior_cdf(x: float, sigma: float = 0.1) -> np.ndarray:
    posterior_pdf = get_posterior_pdf(x, sigma)
    return lambda theta: scipy.integrate.quad(
        posterior_pdf, 0, theta, epsabs=1e-9, epsrel=1e-9
    )[0]


@lru_cache
def get_posterior_ppf(
    x: float,
    sigma: float = 0.1,
    n_interpolation_quantiles: int = 5000,
    epsilon: float = 1e-10,
    n_jobs: int = None,
) -> np.ndarray:
    posterior_pdf = get_posterior_pdf(x, sigma)
    posterior_cdf = get_posterior_cdf(x, sigma)

    def posterior_ppf(q: float):
        return scipy.optimize.root_scalar(
            lambda guess: posterior_cdf(guess) - q,
            bracket=(0, 1),
            fprime=lambda guess: posterior_pdf(guess),
        ).root

    quantiles = np.linspace(0 + epsilon, 1 - epsilon, n_interpolation_quantiles)
    with parallel_backend("loky", inner_max_num_threads=1):
        ppf_points = Parallel(n_jobs=n_jobs)(
            delayed(posterior_ppf)(quantile) for quantile in quantiles.tolist()
        )
    ppf_points = np.asarray(ppf_points)
    return partial(np.interp, xp=quantiles, fp=ppf_points)


def sample_posterior(n: int, posterior_ppf: Callable) -> np.ndarray:
    quantiles = np.random.rand(n)
    return np.asarray([posterior_ppf(quantile) for quantile in quantiles])


class EggBoxPosterior(Distribution):
    arg_constraints = {"observation": constraints.real, "scale": constraints.positive}
    support = constraints.unit_interval
    has_rsample = False

    def __init__(self, observation, scale, validate_args=None):
        observation, scale, low, high = broadcast_all(observation, scale, 0.0, 1.0)
        self.noise = Normal(loc=observation, scale=scale)
        self.bound = Uniform(low=low, high=high)

        self._custom_posterior_ppf = None

        if isinstance(observation, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.observation.size()
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @property
    def observation(self):
        return self.noise.loc

    @property
    def scale(self):
        return self.noise.scale

    @property
    def low(self):
        return self.bound.low

    @property
    def high(self):
        return self.bound.high

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(EggBoxPosterior, _instance)
        batch_shape = torch.Size(batch_shape)
        new.noise = new.noise.expand(batch_shape)
        new.bound = new.bound.expand(batch_shape)
        super(EggBoxPosterior, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def make_posterior_ppf(self, n_interpolation_quantiles, epsilon, n_jobs):
        x = self.observation.detach().flatten().cpu()[0].item()
        sigma = self.scale.detach().flatten().cpu()[0].item()
        self._custom_posterior_ppf = get_posterior_ppf(
            x, sigma, n_interpolation_quantiles, epsilon, n_jobs
        )

    @cached_property
    def posterior_ppf(self):
        if self._custom_posterior_ppf is None:
            x = self.observation.detach().flatten().cpu()[0].item()
            sigma = self.scale.detach().flatten().cpu()[0].item()
            return get_posterior_ppf(x, sigma)
        else:
            return self._custom_posterior_ppf

    def sample(self, sample_shape=torch.Size()):
        # TODO: It would be better if this function depended only on self.log_prob
        shape = self._extended_shape(sample_shape)
        if self._custom_posterior_ppf is None:
            print(
                "It is assumed that the observation & scale is exactly the same in every batch_shape and event_shape dimension."
            )
        else:
            print("Using custom ppf.")
        n = math.prod(shape)
        samples = sample_posterior(n, self.posterior_ppf).reshape(shape)
        return torch.from_numpy(samples).to(dtype=torch.get_default_dtype())

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        transformed = g(value)
        return self.noise.log_prob(transformed) + self.bound.log_prob(value)

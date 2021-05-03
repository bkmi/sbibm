# pylint: disable=no-member, not-callable
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyro
import scipy.stats
import torch
from pyro import distributions as pdist

from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.utils.io import get_tensor_from_csv


class TruncNorm(object):
    def __init__(
        self,
        mean: torch.Tensor,
        low: float,
        high: float,
        std: Union[float, torch.Tensor],
    ):
        self.mean = self._1d_tensor_to_list(mean)
        self.dim = len(self.mean)

        if isinstance(std, torch.Tensor):
            self.std = self._1d_tensor_to_list(std)
        elif isinstance(std, float):
            self.std = [std] * self.dim
        else:
            raise NotImplementedError
        assert self.dim == len(self.std)

        self.a = [(low - x) / s for x, s in zip(self.mean, self.std)]
        self.b = [(high - x) / s for x, s in zip(self.mean, self.std)]
        assert self.dim == len(self.a)
        assert self.dim == len(self.b)

        self.dists = [
            scipy.stats.truncnorm(a=a, b=b, loc=m, scale=s)
            for a, b, m, s in zip(self.a, self.b, self.mean, self.std)
        ]
        assert self.dim == len(self.dists)

    @staticmethod
    def _1d_tensor_to_list(tensor: torch.Tensor) -> List[float]:
        if tensor.ndim == 1:
            tensor = tensor.tolist()
        elif tensor.ndim == 2:
            assert tensor.size(0) == 1
            tensor = tensor[0].tolist()
        else:
            raise NotImplementedError
        return tensor

    def sample(self) -> torch.Tensor:
        sample = [d.rvs() for d in self.dists]
        return torch.tensor(sample).reshape(1, -1).to(torch.float32)

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        sample_list = self._1d_tensor_to_list(sample)
        log_prob = sum([d.logpdf(x) for d, x in zip(self.dists, sample_list)])
        print(log_prob)
        return torch.tensor(log_prob, device=sample.device, dtype=sample.dtype)


class GaussianLinearUniform(Task):
    def __init__(
        self, dim: int = 10, prior_bound: float = 1.0, simulator_scale: float = 0.1
    ):
        """Gaussian Linear Uniform

        Inference of mean under uniform prior.

        Args:
            dim: Dimensionality of parameters and data.
            prior_bound: Prior is uniform in [-prior_bound, +prior_bound].
            simulator_scale: Variance of noise in simulator.
        """
        super().__init__(
            dim_parameters=dim,
            dim_data=dim,
            name=Path(__file__).parent.name,
            name_display="Gaussian Linear Uniform",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
        )

        self.prior_params = {
            "low": -prior_bound * torch.ones((self.dim_parameters,)),
            "high": +prior_bound * torch.ones((self.dim_parameters,)),
        }

        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)

        self.simulator_scale = simulator_scale
        self.simulator_params = {
            "precision_matrix": torch.inverse(
                simulator_scale * torch.eye(self.dim_parameters),
            )
        }

    def get_param_limits(self) -> torch.Tensor:
        return torch.tensor(self.dim_parameters * [[-1, 1]])

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    @staticmethod
    def g(parameters: torch.Tensor) -> torch.Tensor:
        return parameters

    def get_additive_noise(self, key) -> Callable:
        keytype = type(key)

        def noise(simulation: Dict[keytype, np.array], *args):
            x = simulation[key]
            x = x + np.random.randn(*x.shape) * np.sqrt(self.simulator_scale)
            return dict(key=x)

        return noise

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """

        def simulator(parameters):
            return pyro.sample(
                "data",
                pdist.MultivariateNormal(
                    loc=parameters,
                    precision_matrix=self.simulator_params["precision_matrix"],
                ),
            )

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def get_observation(self, num_observation: int) -> torch.Tensor:
        """Get observed data for a given observation number"""
        path = (
            self.path
            / "files"
            / f"dim_{self.dim_parameters:04d}"
            / f"scale_{self.simulator_scale:.0e}"
            / f"num_observation_{num_observation}"
            / "observation.csv"
        )
        return get_tensor_from_csv(path)

    def get_reference_posterior_samples(self, num_observation: int) -> torch.Tensor:
        """Get reference posterior samples for a given observation number"""
        path = (
            self.path
            / "files"
            / f"dim_{self.dim_parameters:04d}"
            / f"scale_{self.simulator_scale:.0e}"
            / f"num_observation_{num_observation}"
            / "reference_posterior_samples.csv.bz2"
        )
        return get_tensor_from_csv(path)

    def get_true_parameters(self, num_observation: int) -> torch.Tensor:
        """Get true parameters (parameters that generated the data) for a given observation number"""
        path = (
            self.path
            / "files"
            / f"dim_{self.dim_parameters:04d}"
            / f"scale_{self.simulator_scale:.0e}"
            / f"num_observation_{num_observation}"
            / "true_parameters.csv"
        )
        return get_tensor_from_csv(path)

    def _get_observation_seed(self, num_observation: int) -> int:
        """Get observation seed for a given observation number"""
        path = (
            self.path
            / "files"
            / f"dim_{self.dim_parameters:04d}"
            / f"scale_{self.simulator_scale:.0e}"
            / f"num_observation_{num_observation}"
            / "observation_seed.csv"
        )
        return int(pd.read_csv(path)["observation_seed"][0])

    def _save_observation(self, num_observation: int, observation: torch.Tensor):
        """Save observed data for a given observation number"""
        path = (
            self.path
            / "files"
            / f"dim_{self.dim_parameters:04d}"
            / f"scale_{self.simulator_scale:.0e}"
            / f"num_observation_{num_observation}"
            / "observation.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_data(path, observation)

    def _save_observation_seed(self, num_observation: int, observation_seed: int):
        """Save observation seed for a given observation number"""
        path = (
            self.path
            / "files"
            / f"dim_{self.dim_parameters:04d}"
            / f"scale_{self.simulator_scale:.0e}"
            / f"num_observation_{num_observation}"
            / "observation_seed.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [[int(observation_seed), int(num_observation)]],
            columns=["observation_seed", "num_observation"],
        ).to_csv(path, index=False)

    def _save_reference_posterior_samples(
        self, num_observation: int, reference_posterior_samples: torch.Tensor
    ):
        """Save reference posterior samples for a given observation number"""
        path = (
            self.path
            / "files"
            / f"dim_{self.dim_parameters:04d}"
            / f"scale_{self.simulator_scale:.0e}"
            / f"num_observation_{num_observation}"
            / "reference_posterior_samples.csv.bz2"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_parameters(path, reference_posterior_samples)

    def _save_true_parameters(
        self, num_observation: int, true_parameters: torch.Tensor
    ):
        """Save true parameters (parameters that generated the data) for a given observation number"""
        path = (
            self.path
            / "files"
            / f"dim_{self.dim_parameters:04d}"
            / f"scale_{self.simulator_scale:.0e}"
            / f"num_observation_{num_observation}"
            / "true_parameters.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_parameters(path, true_parameters)

    # def _sample_reference_posterior(
    #     self,
    #     num_samples: int,
    #     num_observation: Optional[int] = None,
    #     observation: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     """Sample reference posterior for given observation

    #     Uses closed form solution with rejection sampling

    #     Args:
    #         num_samples: Number of samples to generate
    #         num_observation: Observation number
    #         observation: Instead of passing an observation number, an observation may be
    #             passed directly

    #     Returns:
    #         Samples from reference posterior
    #     """
    #     assert not (num_observation is None and observation is None)
    #     assert not (num_observation is not None and observation is not None)

    #     if num_observation is not None:
    #         observation = self.get_observation(num_observation=num_observation)

    #     log = logging.getLogger(__name__)

    #     reference_posterior_samples = []

    # sampling_dist = pdist.MultivariateNormal(
    #     loc=observation, precision_matrix=self.simulator_params["precision_matrix"],
    # )

    #     # Reject samples outside of prior bounds
    #     counter = 0
    #     while len(reference_posterior_samples) < num_samples:
    #         counter += 1
    #         sample = sampling_dist.sample()
    #         if not torch.isinf(self.prior_dist.log_prob(sample).sum()):
    #             reference_posterior_samples.append(sample)

    #     reference_posterior_samples = torch.cat(reference_posterior_samples)
    #     acceptance_rate = float(num_samples / counter)

    #     log.info(
    #         f"Acceptance rate for observation {num_observation}: {acceptance_rate}"
    #     )

    #     return reference_posterior_samples

    def _get_scipy_truncnorm(
        self, observation: torch.Tensor
    ) -> scipy.stats.rv_continuous:
        low = self.prior_params["low"][0].numpy()
        high = self.prior_params["high"][0].numpy()
        std = np.sqrt(self.simulator_scale)
        return TruncNorm(
            mean=observation,
            low=low,
            high=high,
            std=std,
        )

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation

        Uses closed form solution using a truncated normal

        Args:
            num_samples: Number of samples to generate
            num_observation: Observation number
            observation: Instead of passing an observation number, an observation may be
                passed directly

        Returns:
            Samples from reference posterior
        """
        assert not (num_observation is None and observation is None)
        assert not (num_observation is not None and observation is not None)

        if num_observation is not None:
            observation = self.get_observation(num_observation=num_observation)

        log = logging.getLogger(__name__)

        reference_posterior_samples = []

        sampling_dist = self._get_scipy_truncnorm(observation)

        # Reject samples outside of prior bounds, still required even with trunc norm
        counter = 0
        while len(reference_posterior_samples) < num_samples:
            counter += 1
            sample = sampling_dist.sample()
            if not torch.isinf(self.prior_dist.log_prob(sample).sum()):
                reference_posterior_samples.append(sample)

        reference_posterior_samples = torch.cat(reference_posterior_samples)
        acceptance_rate = float(num_samples / counter)

        log.info(
            f"Acceptance rate for observation {num_observation}: {acceptance_rate}"
        )

        return reference_posterior_samples


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--simulator_scale", type=float, default=0.1)
    args = parser.parse_args()

    task = GaussianLinearUniform(
        dim=args.dim,
        simulator_scale=args.simulator_scale,
    )
    task._setup(n_jobs=args.n_jobs)

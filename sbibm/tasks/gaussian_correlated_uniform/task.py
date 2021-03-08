import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import pyro
import scipy.stats
import torch
from pyro import distributions as pdist

from sbibm.tasks.gaussian_linear_uniform.task import TruncNorm
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.utils.io import get_tensor_from_csv


def get_correlated_covariance_matrix(dim: int, correlation: float) -> torch.Tensor:
    covariance_matrix = torch.eye(dim)
    covariance_matrix[torch.triu_indices(dim, dim, 1).unbind()] = correlation
    covariance_matrix[torch.tril_indices(dim, dim, -1).unbind()] = correlation
    return covariance_matrix


def get_random_rotation(dim: int, device=None, dtype=None) -> torch.Tensor:
    return torch.from_numpy(scipy.stats.special_ortho_group(dim).rvs()).to(
        device=device, dtype=dtype
    )


def rotate_matrix(matrix: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return rotation @ matrix @ rotation.T


def normalize_covariance_matrix(covariance_matrix: torch.Tensor) -> torch.Tensor:
    return covariance_matrix / torch.trace(covariance_matrix)


def get_rotated_normalized_covariance_matrix(
    dim: int, correlation: float
) -> torch.Tensor:
    covariance_matrix = get_correlated_covariance_matrix(dim, correlation)
    rotation = get_random_rotation(
        dim, device=covariance_matrix.device, dtype=covariance_matrix.dtype
    )

    normed_covariance_matrix = normalize_covariance_matrix(covariance_matrix)
    return rotate_matrix(normed_covariance_matrix, rotation)


class GaussianCorrelatedUniform(Task):
    def __init__(
        self,
        dim: int = 10,
        prior_bound: float = 1.0,
        simulator_scale: float = 0.1,
        simulator_correlation: float = 0.9,
    ):
        """Gaussian Linear Uniform

        Inference of mean under uniform prior.

        Args:
            dim: Dimensionality of parameters and data.
            prior_bound: Prior is uniform in [-prior_bound, +prior_bound].
            simulator_scale: Standard deviation of noise in simulator.
            simulator_correlation: off diagonal value in correlation matrix.
        """
        super().__init__(
            dim_parameters=dim,
            dim_data=dim,
            name=Path(__file__).parent.name,
            name_display="Gaussian Correlated Uniform",
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

        self.simulator_correlation = simulator_correlation
        self.simulator_scale = simulator_scale
        self.simulator_params = {
            "precision_matrix": torch.inverse(
                simulator_scale
                * get_rotated_normalized_covariance_matrix(
                    self.dim_parameters, simulator_correlation
                ),
            )
        }

    def get_param_limits(self) -> torch.Tensor:
        return torch.tensor(self.dim_parameters * [[-1, 1]])

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

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
            / f"correlation_{self.simulator_correlation:.0e}"
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
            / f"correlation_{self.simulator_correlation:.0e}"
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
            / f"correlation_{self.simulator_correlation:.0e}"
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
            / f"correlation_{self.simulator_correlation:.0e}"
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
            / f"correlation_{self.simulator_correlation:.0e}"
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
            / f"correlation_{self.simulator_correlation:.0e}"
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
            / f"correlation_{self.simulator_correlation:.0e}"
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
            / f"correlation_{self.simulator_correlation:.0e}"
            / f"num_observation_{num_observation}"
            / "true_parameters.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_parameters(path, true_parameters)

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation

        Uses closed form solution with rejection sampling

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

        sampling_dist = pdist.MultivariateNormal(
            loc=observation,
            precision_matrix=self.simulator_params["precision_matrix"],
        )

        # Reject samples outside of prior bounds
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

    #########################################
    # below ought to work in higher dim but
    # assigns zero probability to an observation
    # leading to an infinite domination_constant
    #########################################

    # def _get_scipy_truncnorm(
    #     self, observation: torch.Tensor
    # ) -> scipy.stats.rv_continuous:
    #     low = self.prior_params["low"][0].numpy()
    #     high = self.prior_params["high"][0].numpy()

    #     # This seems to underestimate in some dimensions
    #     evals, _ = torch.symeig(self.simulator_params["precision_matrix"])
    #     scale = 1 / evals

    #     # scale = 1 / torch.trace(self.simulator_params["precision_matrix"]).numpy()
    #     return TruncNorm(
    #         mean=observation,
    #         low=low,
    #         high=high,
    #         scale=scale,
    #     )

    # def _sample_reference_posterior(
    #     self,
    #     num_samples: int,
    #     num_observation: Optional[int] = None,
    #     observation: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     """Sample reference posterior for given observation

    #     Uses closed form solution using a truncated normal

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

    #     sampling_dist = self._get_scipy_truncnorm(observation)
    #     posterior_dist = pdist.MultivariateNormal(
    #         loc=observation,
    #         precision_matrix=self.simulator_params["precision_matrix"],
    #     )
    #     domination_constant = posterior_dist.log_prob(observation).item() - sampling_dist.log_prob(observation)
    #     # breakpoint()
    #     # assert domination_constant >= 0.0
    #     assert not torch.isinf(domination_constant)

    #     counter = 0
    #     while len(reference_posterior_samples) < num_samples:
    #         counter += 1
    #         sample = sampling_dist.sample()
    #         if not torch.isinf(self.prior_dist.log_prob(sample).sum()):
    #             # Reject samples outside of prior bounds
    #             chance = posterior_dist.log_prob(sample) - (domination_constant + sampling_dist.log_prob(sample))
    #             # print(chance)
    #             # print(torch.rand(1).log().item())
    #             breakpoint()
    #             if torch.rand(1).log().item() < chance:
    #                 breakpoint()
    #                 reference_posterior_samples.append(sample)

    #     reference_posterior_samples = torch.cat(reference_posterior_samples)
    #     acceptance_rate = float(num_samples / counter)

    #     log.info(
    #         f"Acceptance rate for observation {num_observation}: {acceptance_rate}"
    #     )

    #     return reference_posterior_samples


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--simulator_scale", type=float, default=0.1)
    parser.add_argument("--simulator_correlation", type=float, default=0.9)
    args = parser.parse_args()

    task = GaussianCorrelatedUniform(
        dim=args.dim,
        simulator_scale=args.simulator_scale,
        simulator_correlation=args.simulator_correlation,
    )
    task._setup(n_jobs=args.n_jobs)

import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import pyro
import scipy.stats
import torch
from pyro import distributions as pdist

from sbibm.tasks.eggbox.posterior import EggBoxPosterior
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.utils.io import get_tensor_from_csv


class EggBox(Task):
    def __init__(
        self,
        dim: int = 10,
        simulator_scale: float = 0.1,
    ):
        """Egg Box

        Inference of transformed mean under uniform prior.

        Args:
            dim: Dimensionality of parameters and data.
            simulator_scale: Standard deviation of noise in simulator.
        """
        super().__init__(
            dim_parameters=dim,
            dim_data=dim,
            name=Path(__file__).parent.name,
            name_display="Egg Box",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
        )

        self.prior_params = {
            "low": torch.zeros((self.dim_parameters,)),
            "high": torch.ones((self.dim_parameters,)),
        }

        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)

        self.simulator_scale = simulator_scale
        self.simulator_params = {
            "precision_matrix": torch.inverse(
                self.simulator_scale * torch.eye(self.dim_parameters)
            )
        }

    def get_param_limits(self) -> torch.Tensor:
        return torch.stack(
            [self.prior_params["low"], self.prior_params["high"]], dim=-1
        )

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    @staticmethod
    def g(parameters: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * parameters) ** 2

    def get_additive_noise(self, key) -> Callable:
        keytype = type(key)

        def noise(simulation: Dict[keytype, np.array], *args):
            x = scipy.stats.multivariate_normal(
                mean=simulation[key],
                cov=torch.inverse(self.simulator_params["precision_matrix"])
                .detach()
                .cpu()
                .numpy(),
            ).rvs()
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
            return self.g(parameters) + pyro.sample(
                "data",
                pdist.MultivariateNormal(
                    loc=torch.zeros_like(parameters),
                    precision_matrix=self.simulator_params["precision_matrix"],
                ),
            )

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _get_base_path(self, num_observation: int) -> str:
        return (
            self.path
            / "files"
            / f"scale_{self.simulator_scale:.0e}"
            / f"num_observation_{num_observation}"
        )

    def get_observation(self, num_observation: int) -> torch.Tensor:
        """Get observed data for a given observation number"""
        path = self._get_base_path(num_observation) / "observation.csv"
        return get_tensor_from_csv(path)

    def get_reference_posterior_samples(self, num_observation: int) -> torch.Tensor:
        """Get reference posterior samples for a given observation number"""
        path = (
            self._get_base_path(num_observation) / "reference_posterior_samples.csv.bz2"
        )
        return get_tensor_from_csv(path)

    def get_true_parameters(self, num_observation: int) -> torch.Tensor:
        """Get true parameters (parameters that generated the data) for a given observation number"""
        path = self._get_base_path(num_observation) / "true_parameters.csv"
        return get_tensor_from_csv(path)

    def _get_observation_seed(self, num_observation: int) -> int:
        """Get observation seed for a given observation number"""
        path = self._get_base_path(num_observation) / "observation_seed.csv"
        return int(pd.read_csv(path)["observation_seed"][0])

    def _save_observation(self, num_observation: int, observation: torch.Tensor):
        """Save observed data for a given observation number"""
        path = self._get_base_path(num_observation) / "observation.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_data(path, observation)

    def _save_observation_seed(self, num_observation: int, observation_seed: int):
        """Save observation seed for a given observation number"""
        path = self._get_base_path(num_observation) / "observation_seed.csv"
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
            self._get_base_path(num_observation) / "reference_posterior_samples.csv.bz2"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_parameters(path, reference_posterior_samples)

    def _save_true_parameters(
        self, num_observation: int, true_parameters: torch.Tensor
    ):
        """Save true parameters (parameters that generated the data) for a given observation number"""
        path = self._get_base_path(num_observation) / "true_parameters.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_parameters(path, true_parameters)

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
        n_interpolation_quantiles: int = 5000,
        epsilon: float = 1e-10,
        n_jobs: int = -1,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation

        Uses custom inverse transform sampling method.

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

        observation = observation.squeeze()
        assert observation.ndim == 1
        posterior = EggBoxPosterior(observation, self.simulator_scale)
        posterior.make_posterior_ppf(n_interpolation_quantiles, epsilon, n_jobs)
        return posterior.sample((num_samples,))

    def _setup(self, n_jobs: int = -1, create_reference: bool = True, **kwargs: Any):
        """Setup the task: generate observations and reference posterior samples

        In most cases, you don't need to execute this method, since its results are stored to disk.

        Re-executing will overwrite existing files.

        Args:
            n_jobs: Number of to use for Joblib
            create_reference: If False, skips reference creation
        """
        from joblib import Parallel, delayed  # noqa: F401

        def run(num_observation, observation_seed, **kwargs):
            np.random.seed(observation_seed)
            torch.manual_seed(observation_seed)
            self._save_observation_seed(num_observation, observation_seed)

            prior = self.get_prior()
            if num_observation == 1:
                true_parameters = torch.ones_like(prior(num_samples=1)) * 0.25
            else:
                true_parameters = prior(num_samples=1) * 0.25
            self._save_true_parameters(num_observation, true_parameters)

            simulator = self.get_simulator()
            if num_observation == 1:
                observation = true_parameters.clone()
            else:
                observation = simulator(true_parameters)
            self._save_observation(num_observation, observation)

            if create_reference:
                reference_posterior_samples = self._sample_reference_posterior(
                    num_observation=num_observation,
                    num_samples=self.num_reference_posterior_samples,
                    **kwargs,
                )
                num_unique = torch.unique(reference_posterior_samples, dim=0).shape[0]
                assert num_unique == self.num_reference_posterior_samples
                self._save_reference_posterior_samples(
                    num_observation,
                    reference_posterior_samples,
                )

        print("Only makes 1 samples for now, too parallel.")
        num_observation = 1
        observation_seed = self.observation_seeds[0]
        run(num_observation, observation_seed)

        # Parallel(n_jobs=n_jobs, verbose=50, backend="loky")(
        #     delayed(run)(num_observation, observation_seed, **kwargs)
        #     for num_observation, observation_seed in enumerate(
        #         self.observation_seeds, start=1
        #     )
        # )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--simulator_scale", type=float, default=0.1)
    parser.add_argument("--unit_cube_epsilon", type=float, default=1e-10)
    parser.add_argument("--n_interpolation_quantiles", type=int, default=5000)
    args = parser.parse_args()

    task = EggBox(
        dim=args.dim,
        simulator_scale=args.simulator_scale,
    )
    task._setup(
        n_jobs=args.n_jobs,
        n_interpolation_quantiles=args.n_interpolation_quantiles,
        epsilon=args.unit_cube_epsilon,
    )

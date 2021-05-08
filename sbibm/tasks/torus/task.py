from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import pyro
import scipy
import scipy.stats
import torch
from pyro import distributions as pdist
from toolz import compose

from sbibm.tasks.rejectionsampling import rejection_sample
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.utils.io import get_tensor_from_csv


class Torus(Task):
    def __init__(
        self,
        simulator_scale: tuple = (0.03, 0.005, 0.2),
    ):
        """Torus

        Inference of parameters under uniform prior.

        Args:
            simulator_scale: Diagonal standard deviation of noise in simulator. Default (0.03, 0.005, 0.2).
        """
        super().__init__(
            dim_parameters=3,
            dim_data=3,
            name=Path(__file__).parent.name,
            name_display="Torus",
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

        self.simulator_scale = torch.tensor(simulator_scale)
        self.simulator_std = self.simulator_scale
        self.simulator_var = self.simulator_scale ** 2
        self.simulator_params = {
            "covariance_matrix": torch.diag(self.simulator_var),
            "precision_matrix": torch.inverse(torch.diag(self.simulator_var)),
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
        a = parameters[..., 0]
        b = parameters[..., 1]
        if isinstance(parameters, torch.Tensor):
            c = torch.sqrt((a - 0.6) ** 2 + (b - 0.8) ** 2)
            return torch.stack([parameters[..., 0], c, parameters[..., 2]], dim=-1)
        else:
            c = np.sqrt((a - 0.6) ** 2 + (b - 0.8) ** 2)
            return np.stack([parameters[..., 0], c, parameters[..., 2]], axis=-1)

    def get_additive_noise(self, key) -> Callable:
        keytype = type(key)

        def noise(simulation: Dict[keytype, np.array], *args):
            x = simulation[key]
            x = (
                x
                + np.random.randn(*x.shape) * self.simulator_std.detach().cpu().numpy()
            )
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
                    loc=self.g(parameters),
                    precision_matrix=self.simulator_params["precision_matrix"],
                ),
            )

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _get_base_path(self, num_observation: int) -> str:
        return (
            self.path
            / "files"
            / ("scale_" + "_".join([f"{s:.3f}" for s in self.simulator_scale]))
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

    def _get_unnormalized_posterior_logpdf(self, observation) -> Callable:
        normal = scipy.stats.multivariate_normal(
            mean=observation,
            cov=self.simulator_params["covariance_matrix"].detach().cpu().numpy(),
        )
        return compose(normal.logpdf, self.g)

    def _get_max_unnormalized_posterior_logpdf(self, observation):
        unnormalized_posterior_logpdf = self._get_unnormalized_posterior_logpdf(
            observation,
        )

        # # If there is a num_observation
        # argmax = scipy.optimize.minimize(
        #     lambda x: -unnormalized_posterior_logpdf(x),
        #     x0=self.get_true_parameters().numpy(),
        #     jac='3-point',
        #     bounds=self.get_param_limits().numpy(),
        # )
        # if argmax.success == False:
        #     print("logmax did not converge.")
        # return unnormalized_posterior_logpdf(argmax.x), argmax

        xx, yy, zz = np.mgrid[0:1:0.01, 0:1:0.01, 0:1:0.01]
        thetas = np.stack([xx, yy, zz], axis=-1)
        probs = unnormalized_posterior_logpdf(thetas)

        indmax = np.argmax(probs)
        indmax = np.unravel_index(indmax, probs.shape)
        argmax = thetas[indmax]
        argmax = scipy.optimize.minimize(
            lambda x: -unnormalized_posterior_logpdf(x),
            x0=argmax,
            jac="3-point",
            bounds=self.get_param_limits().numpy(),
        )
        if not argmax.success:
            argmax = scipy.optimize.minimize(
                lambda x: -unnormalized_posterior_logpdf(x),
                x0=argmax.x,
                hess="3-point",
                bounds=self.get_param_limits().numpy(),
            )
        if not argmax.success:
            print("logmax did not converge.")
        return unnormalized_posterior_logpdf(argmax.x), argmax

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

        sample_fn = lambda x: np.random.rand(x, 3)  # noqa: E731
        logpdf_sampling = lambda x: np.zeros(len(x))  # noqa: E731
        logpdf_target = self._get_unnormalized_posterior_logpdf(
            observation.squeeze().numpy()
        )
        logmaxratio, _ = self._get_max_unnormalized_posterior_logpdf(
            observation.squeeze().numpy()  # posterior term
        )  # flat prior term, subtract zero

        reference_posterior_samples = rejection_sample(
            n=num_samples,
            sample_fn=sample_fn,
            logpdf_sampling=logpdf_sampling,
            logpdf_target=logpdf_target,
            logmaxratio=logmaxratio,
            maxiter=500_000,
        )

        if not len(reference_posterior_samples) == num_samples:
            print(num_observation, len(reference_posterior_samples))

        return torch.from_numpy(reference_posterior_samples).to(torch.float32)

    def _setup(self, n_jobs: int = -1, create_reference: bool = True, **kwargs: Any):
        """Setup the task: generate observations and reference posterior samples

        In most cases, you don't need to execute this method, since its results are stored to disk.

        Re-executing will overwrite existing files.

        Args:
            n_jobs: Number of to use for Joblib
            create_reference: If False, skips reference creation
        """
        from joblib import Parallel, delayed

        def run(num_observation, observation_seed, **kwargs):
            np.random.seed(observation_seed)
            torch.manual_seed(observation_seed)
            self._save_observation_seed(num_observation, observation_seed)

            prior = self.get_prior()
            true_parameters = compose(torch.atleast_2d, torch.tensor)([0.57, 0.8, 1.0])
            self._save_true_parameters(num_observation, true_parameters)

            simulator = self.get_simulator()
            if num_observation == 1:
                observation = self.g(true_parameters)
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

        Parallel(n_jobs=n_jobs, verbose=50, backend="loky")(
            delayed(run)(num_observation, observation_seed, **kwargs)
            for num_observation, observation_seed in enumerate(
                self.observation_seeds, start=1
            )
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    task = Torus(
        # simulator_scale=args.simulator_scale,
    )
    task._setup(n_jobs=args.n_jobs)

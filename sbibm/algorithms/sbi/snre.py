import logging
import math
from typing import Any, Dict, Optional, Tuple

import cnre
import cnre.data.presampled
import torch
from sbi import inference as inference
from sbi.utils.get_nn_models import classifier_nn

from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.tasks.task import Task


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    training_samples_root: Optional[str] = None,
    observation: Optional[torch.Tensor] = None,
    num_rounds: int = 10,
    neural_net: str = "resnet",
    hidden_features: int = 50,
    num_blocks: int = 2,
    use_batch_norm: bool = True,
    simulation_batch_size: int = 1000,
    training_batch_size: int = 10000,
    num_atoms: int = 10,
    automatic_transforms_enabled: bool = True,
    sample_with: str = "mcmc",
    mcmc_method: str = "slice_np_vectorized",
    mcmc_parameters: Dict[str, Any] = {
        "num_chains": 100,
        "thin": 10,
        "warmup_steps": 100,
        "init_strategy": "sir",
    },
    z_score_x: bool = True,
    z_score_theta: bool = True,
    variant: str = "B",
    max_num_epochs: Optional[int] = None,
    stop_after_epochs: int = 2 ** 31 - 1,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs (S)NRE from `sbi`

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_simulations: Simulation budget
        num_rounds: Number of rounds
        neural_net: Neural network to use, one of linear / mlp / resnet
        hidden_features: Number of hidden features in network
        simulation_batch_size: Batch size for simulator
        training_batch_size: Batch size for training network
        num_atoms: Number of atoms, -1 means same as `training_batch_size`
        automatic_transforms_enabled: Whether to enable automatic transforms
        mcmc_method: MCMC method
        mcmc_parameters: MCMC parameters
        z_score_x: Whether to z-score x
        z_score_theta: Whether to z-score theta
        variant: Can be used to switch between SNRE-A (AALR) and -B (SRE)
        max_num_epochs: Maximum number of epochs

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = logging.getLogger(__name__)

    if num_rounds == 1:
        log.info(f"Running NRE")
        num_simulations_per_round = num_simulations
    else:
        log.info(f"Running SNRE")
        num_simulations_per_round = math.floor(num_simulations / num_rounds)

    if simulation_batch_size > num_simulations_per_round:
        simulation_batch_size = num_simulations_per_round
        log.warn("Reduced simulation_batch_size to num_simulation_per_round")

    if training_batch_size > num_simulations_per_round:
        training_batch_size = num_simulations_per_round
        log.warn("Reduced training_batch_size to num_simulation_per_round")

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

    simulator = task.get_simulator(max_calls=num_simulations)

    transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]
    if automatic_transforms_enabled:
        prior = wrap_prior_dist(prior, transforms)
        simulator = wrap_simulator_fn(simulator, transforms)

    classifier = classifier_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        use_batch_norm=use_batch_norm,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )
    if variant == "A":
        inference_class = inference.SNRE_A
        inference_method_kwargs = {}
    elif variant == "B":
        inference_class = inference.SNRE_B
        inference_method_kwargs = {"num_atoms": num_atoms}
    else:
        raise NotImplementedError

    inference_method = inference_class(classifier=classifier, prior=prior)

    posteriors = []
    proposal = prior
    mcmc_parameters["warmup_steps"] = 25
    # mcmc_parameters["enable_transform"] = False  # NOTE: Disable `sbi` auto-transforms, since `sbibm` does its own

    for r in range(num_rounds):
        if training_samples_root is None:
            log.info(f"simulating round {r} training samples")
            theta, x = inference.simulate_for_sbi(
                simulator,
                proposal,
                num_simulations=num_simulations_per_round,
                simulation_batch_size=simulation_batch_size,
            )
        elif training_samples_root is not None and num_rounds == 1:
            log.info(f"loading training samples from: {training_samples_root}")
            theta, x = cnre.data.presampled.load_training_samples(
                task.name, num_simulations, training_samples_root
            )
            if automatic_transforms_enabled:
                theta = transforms(theta)
        else:
            raise NotImplementedError(
                "Cannot load preloaded training samples and learn sequentially."
            )

        density_estimator = inference_method.append_simulations(
            theta, x, from_round=r
        ).train(
            training_batch_size=training_batch_size,
            retrain_from_scratch=False,
            discard_prior_samples=False,
            show_train_summary=True,
            max_num_epochs=max_num_epochs,
            stop_after_epochs=stop_after_epochs,
            **inference_method_kwargs,
        )
        if r > 1:
            mcmc_parameters["init_strategy"] = "latest_sample"

        posterior = inference_method.build_posterior(
            density_estimator,
            sample_with=sample_with,
            mcmc_method=mcmc_method,
            mcmc_parameters=mcmc_parameters,
            enable_transform=False,  # NOTE: Disable `sbi` auto-transforms, since `sbibm` does its own
        )
        # Copy hyperparameters, e.g., mcmc_init_samples for "latest_sample" strategy.
        if r > 0:
            posterior.copy_hyperparameters_from(posteriors[-1])
        proposal = posterior.set_default_x(observation)
        posteriors.append(posterior)

    posterior = wrap_posterior(posteriors[-1], transforms)

    if training_samples_root is None:
        assert simulator.num_simulations == num_simulations
        checked_num_simulations = simulator.num_simulations
    else:
        checked_num_simulations = num_simulations

    samples = posterior.sample((num_samples,)).detach()


    theta, x, _ = inference_method.get_simulations()
    dataset = torch.utils.data.TensorDataset(theta, x)
    _, valid_loader = inference_method.get_dataloaders(
        dataset,
        training_batch_size,
        validation_fraction=0.1,
        resume_training=True,
    )

    avg_log_ratio = cnre.expected_log_ratio(valid_loader, inference_method._neural_net)

    return samples, checked_num_simulations, None, avg_log_ratio

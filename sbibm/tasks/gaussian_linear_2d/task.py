from pathlib import Path
from sbibm.tasks.gaussian_linear.task import GaussianLinear


class GaussianLinear2d(GaussianLinear):
    def __init__(
        self, dim: int = 2, prior_scale: float = 0.1, simulator_scale: float = 0.1
    ):
        """Gaussian Linear

        Inference of mean under Gaussian prior in 2D

        Args:
            dim: Dimensionality of parameters and data
            prior_scale: Standard deviation of prior
            simulator_scale: Standard deviation of noise in simulator
        """
        super().__init__(dim, prior_scale, simulator_scale)
        self.name = Path(__file__).parent.name
        self.path = Path(__file__).parent.absolute()


if __name__ == "__main__":
    task = GaussianLinear2d()
    task._setup()

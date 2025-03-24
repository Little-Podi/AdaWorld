from lightning.pytorch.cli import LightningCLI

from lam.dataset import LightningPlatformer2D
from lam.model import LAM

cli = LightningCLI(
    LAM,
    LightningPlatformer2D,
    seed_everything_default=32
)

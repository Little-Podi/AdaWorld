from lightning.pytorch.cli import LightningCLI

from lam.dataset import LightningVideoDataset
from lam.model import LAM

cli = LightningCLI(
    LAM,
    LightningVideoDataset,
    seed_everything_default=32
)

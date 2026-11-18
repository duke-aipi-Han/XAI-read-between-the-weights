"""Deterministic independent synthetic datasets."""
import torch

QUADRANTS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def make_split(size: int, seed: int, quadrant_balanced: bool = False):
    """Generate inputs and same-sign labels without changing global RNG state.

    Balanced samples cycle quadrants, so each consecutive group of eight contains
    two examples per quadrant, as required for fixed probe batches.
    """
    if size <= 0 or (quadrant_balanced and size % 4):
        raise ValueError('size must be positive and balanced sizes divisible by four')
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand((size, 4), generator=generator, dtype=torch.float32) * 2 - 1
    if quadrant_balanced:
        signs = torch.tensor(QUADRANTS, dtype=x.dtype).repeat(size // 4, 1)
        # Avoid exact zero so every example belongs to its assigned quadrant.
        x[:, :2] = x[:, :2].abs().clamp_min(torch.finfo(x.dtype).eps) * signs
    y = ((x[:, 0] * x[:, 1]) > 0).to(torch.float32).unsqueeze(1)
    return x, y


def make_splits(config):
    """Build the independently seeded splits specified in the configuration."""
    configs = config['splits']
    seeds = [item['seed'] for item in configs.values()]
    if len(seeds) != len(set(seeds)):
        raise ValueError('dataset splits must use different seeds')
    return {name: make_split(**options) for name, options in configs.items()}

import torch
from araumi.datasets import IsoMoG

def test_import_modules():
    from araumi.datasets import IsoMoG, swiss_roll  # noqa: F401

def test_mog_sample_shape():
    K, d = 3, 2
    means = torch.randn(K, d)
    sigmas = torch.full((K,), 0.5)
    weights = torch.ones(K)
    x = IsoMoG(means, sigmas, weights).sample(128)
    assert x.shape == (128, d)

def test_swiss_roll_shape():
    pass
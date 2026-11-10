import torch

from rectools.models.nn.transformers.negative_sampler import CatalogUniformSampler


def test_catalog_uniform_sampler_default_behavior() -> None:
    batch_dict = {
        "x": torch.tensor([[1, 2]]),
        "y": torch.tensor([[3, 4]]),
    }

    torch.manual_seed(42)
    expected = torch.randint(1, 10, (1, 2, 2))

    torch.manual_seed(42)
    sampler = CatalogUniformSampler(n_negatives=2)
    actual = sampler.get_negatives(batch_dict, lowest_id=1, highest_id=10)

    assert torch.equal(actual, expected)


def test_catalog_uniform_sampler_avoids_positive_collisions(monkeypatch) -> None:
    batch_dict = {
        "x": torch.tensor([[1, 2]]),
        "y": torch.tensor([[3, 0]]),
    }

    samples = iter([
        torch.tensor([[[3, 5], [6, 7]]]),
        torch.tensor([8]),
    ])

    def mock_randint(*args, **kwargs):
        return next(samples)

    monkeypatch.setattr(torch, "randint", mock_randint)

    sampler = CatalogUniformSampler(
        n_negatives=2,
        avoid_positive_collisions=True,
    )

    actual = sampler.get_negatives(
        batch_dict,
        lowest_id=1,
        highest_id=10,
    )

    expected = torch.tensor([[[8, 5], [6, 7]]])

    assert torch.equal(actual, expected)

def test_catalog_uniform_sampler_redraws_only_once(monkeypatch) -> None:
    batch_dict = {
        "x": torch.tensor([[1]]),
        "y": torch.tensor([[3]]),
    }

    samples = iter([
        torch.tensor([[[3, 5]]]),
        torch.tensor([3]),
    ])

    def mock_randint(*args, **kwargs):
        return next(samples)

    monkeypatch.setattr(torch, "randint", mock_randint)

    sampler = CatalogUniformSampler(
        n_negatives=2,
        avoid_positive_collisions=True,
    )

    actual = sampler.get_negatives(
        batch_dict,
        lowest_id=1,
        highest_id=10,
    )

    expected = torch.tensor([[[3, 5]]])

    assert torch.equal(actual, expected)
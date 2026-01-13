import numpy as np

from tinytorch import Tensor
from tinytorch.data import DataLoader, TensorDataset


def test_unit_dataloader():
    print("\n Unit Test: DataLoader...")

    # Create test dataset
    features = Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])  # 5 samples
    labels = Tensor([0, 1, 0, 1, 0])
    dataset = TensorDataset(features, labels)

    # Test basic batching (no shuffle)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Test length calculation
    assert len(loader) == 3, f"Expected 3 batches, got {len(loader)}"  # ceil(5/2) = 3

    batches = list(loader)
    assert len(batches) == 3, f"Expected 3 batches, got {len(batches)}"

    # Test first batch
    batch_features, batch_labels = batches[0]
    assert batch_features.data.shape == (2, 2), (
        f"Wrong batch features shape: {batch_features.data.shape}"
    )
    assert batch_labels.data.shape == (2,), (
        f"Wrong batch labels shape: {batch_labels.data.shape}"
    )

    # Test last batch (should have 1 sample)
    batch_features, batch_labels = batches[2]
    assert batch_features.data.shape == (1, 2), (
        f"Wrong last batch features shape: {batch_features.data.shape}"
    )
    assert batch_labels.data.shape == (1,), (
        f"Wrong last batch labels shape: {batch_labels.data.shape}"
    )

    # Test that data is preserved
    assert np.array_equal(batches[0][0].data[0], [1, 2]), "First sample should be [1,2]"
    assert batches[0][1].data[0] == 0, "First label should be 0"

    # Test shuffling produces different order
    loader_shuffle = DataLoader(dataset, batch_size=5, shuffle=True)
    loader_no_shuffle = DataLoader(dataset, batch_size=5, shuffle=False)

    batch_shuffle = list(loader_shuffle)[0]
    batch_no_shuffle = list(loader_no_shuffle)[0]

    # Note: This might occasionally fail due to random chance, but very unlikely
    # We'll just test that both contain all the original data
    shuffle_features = {tuple(row) for row in batch_shuffle[0].data}
    no_shuffle_features = {tuple(row) for row in batch_no_shuffle[0].data}
    expected_features = {(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)}

    assert shuffle_features == expected_features, "Shuffle should preserve all data"
    assert no_shuffle_features == expected_features, (
        "No shuffle should preserve all data"
    )

    print("✅ DataLoader works correctly!")


def test_unit_dataloader_deterministic():
    print("\n Unit Test: DataLoader Deterministic Shuffling...")

    # Create test dataset
    features = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    labels = Tensor([0, 1, 0, 1])
    dataset = TensorDataset(features, labels)

    # Test that same seed produces same shuffle
    np.random.seed(42)
    loader1 = DataLoader(dataset, batch_size=2, shuffle=True)
    batches1 = list(loader1)

    np.random.seed(42)
    loader2 = DataLoader(dataset, batch_size=2, shuffle=True)
    batches2 = list(loader2)

    # Should produce identical batches with same seed
    for i, (batch1, batch2) in enumerate(zip(batches1, batches2)):
        assert np.array_equal(batch1[0].data, batch2[0].data), (
            f"Batch {i} features should be identical with same seed"
        )
        assert np.array_equal(batch1[1].data, batch2[1].data), (
            f"Batch {i} labels should be identical with same seed"
        )

    # Test that different seeds produce different shuffles
    np.random.seed(42)
    loader3 = DataLoader(dataset, batch_size=2, shuffle=True)
    batches3 = list(loader3)

    np.random.seed(123)  # Different seed
    loader4 = DataLoader(dataset, batch_size=2, shuffle=True)
    batches4 = list(loader4)

    # Should produce different batches with different seeds (very likely)
    different = False
    for batch3, batch4 in zip(batches3, batches4):
        if not np.array_equal(batch3[0].data, batch4[0].data):
            different = True
            break

    assert different, "Different seeds should produce different shuffles"

    print("✅ Deterministic shuffling works correctly!")

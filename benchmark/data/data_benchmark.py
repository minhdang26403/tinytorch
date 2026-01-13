import sys
import time

import numpy as np

from tinytorch import Tensor
from tinytorch.data import DataLoader, TensorDataset


def analyze_dataloader_performance():
    """📊 Analyze DataLoader performance characteristics."""
    print("📊 Analyzing DataLoader Performance...")

    # Create test dataset of varying sizes
    sizes = [1000, 5000, 10000]
    batch_sizes = [16, 64, 256]

    print("\n🔍 Batch Size vs Loading Time:")

    for size in sizes:
        # Create synthetic dataset
        features = Tensor(np.random.randn(size, 100))  # 100 features
        labels = Tensor(np.random.randint(0, 10, size))
        dataset = TensorDataset(features, labels)

        print(f"\nDataset size: {size} samples")

        for batch_size in batch_sizes:
            # Time data loading
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            start_time = time.time()
            batch_count = 0
            for batch in loader:
                batch_count += 1
            end_time = time.time()

            elapsed = end_time - start_time
            throughput = size / elapsed if elapsed > 0 else float("inf")

            print(
                f"  Batch size {batch_size:3d}: {elapsed:.3f}s ({throughput:,.0f} "
                f"samples/sec)"
            )

    # Analyze shuffle overhead
    print("\n🔄 Shuffle Overhead Analysis:")

    dataset_size = 10000
    features = Tensor(np.random.randn(dataset_size, 50))
    labels = Tensor(np.random.randint(0, 5, dataset_size))
    dataset = TensorDataset(features, labels)

    batch_size = 64

    # No shuffle
    loader_no_shuffle = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    start_time = time.time()
    _batches_no_shuffle = list(loader_no_shuffle)  # noqa: F841 (materialize for timing)
    time_no_shuffle = time.time() - start_time

    # With shuffle
    loader_shuffle = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    _batches_shuffle = list(loader_shuffle)  # noqa: F841 (materialize for timing)
    time_shuffle = time.time() - start_time

    shuffle_overhead = ((time_shuffle - time_no_shuffle) / time_no_shuffle) * 100

    print(f"  No shuffle: {time_no_shuffle:.3f}s")
    print(f"  With shuffle: {time_shuffle:.3f}s")
    print(f"  Shuffle overhead: {shuffle_overhead:.1f}%")

    print("\n💡 Key Insights:")
    print("• Larger batch sizes reduce per-sample overhead")
    print("• Shuffle adds minimal overhead for reasonable dataset sizes")
    print("• Memory usage scales linearly with batch size")
    print("🚀 Production tip: Balance batch size with GPU memory limits")


def analyze_memory_usage():
    """📊 Analyze memory usage patterns in data loading."""
    print("\n📊 Analyzing Memory Usage Patterns...")

    # Memory usage estimation
    def estimate_memory_mb(batch_size, feature_size, dtype_bytes=4):
        """Estimate memory usage for a batch."""
        return (batch_size * feature_size * dtype_bytes) / (1024 * 1024)

    print("\n💾 Memory Usage by Batch Configuration:")

    feature_sizes = [784, 3072, 50176]  # MNIST, CIFAR-10, ImageNet-like
    feature_names = ["MNIST (28×28)", "CIFAR-10 (32×32×3)", "ImageNet (224×224×1)"]
    batch_sizes = [1, 32, 128, 512]

    for feature_size, name in zip(feature_sizes, feature_names):
        print(f"\n{name}:")
        for batch_size in batch_sizes:
            memory_mb = estimate_memory_mb(batch_size, feature_size)
            print(f"  Batch {batch_size:3d}: {memory_mb:6.1f} MB")

    print("\n🎯 Memory Trade-offs:")
    print("• Larger batches: More memory, better GPU utilization")
    print("• Smaller batches: Less memory, more noisy gradients")
    print("• Sweet spot: Usually 32-128 depending on model size")

    # Demonstrate actual memory usage with our tensors
    print("\n🔬 Actual Tensor Memory Usage:")

    # Create different sized tensors
    tensor_small = Tensor(np.random.randn(32, 784))  # Small batch
    tensor_large = Tensor(np.random.randn(512, 784))  # Large batch

    # Measure actual memory (data array + object overhead)
    small_bytes = tensor_small.data.nbytes
    large_bytes = tensor_large.data.nbytes

    # Also measure Python object overhead
    small_total = sys.getsizeof(tensor_small.data) + sys.getsizeof(tensor_small)
    large_total = sys.getsizeof(tensor_large.data) + sys.getsizeof(tensor_large)

    print("  Small batch (32×784):")
    print(f"    - Data only: {small_bytes / 1024:.1f} KB")
    print(f"    - With object overhead: {small_total / 1024:.1f} KB")
    print("  Large batch (512×784):")
    print(f"    - Data only: {large_bytes / 1024:.1f} KB")
    print(f"    - With object overhead: {large_total / 1024:.1f} KB")
    print(f"  Ratio: {large_bytes / small_bytes:.1f}× (data scales linearly)")

    print("\n🎯 Memory Optimization Tips:")
    print("• Object overhead becomes negligible with larger batches")
    print("• Use float32 instead of float64 to halve memory usage")
    print("• Consider gradient accumulation for effective larger batches")


def analyze_collation_overhead():
    """📊 Analyze the cost of collating samples into batches."""
    print("\n📊 Analyzing Collation Overhead...")

    # Test different batch sizes to see collation cost
    dataset_size = 1000
    feature_size = 100
    features = Tensor(np.random.randn(dataset_size, feature_size))
    labels = Tensor(np.random.randint(0, 10, dataset_size))
    dataset = TensorDataset(features, labels)

    print("\n⚡ Collation Time by Batch Size:")

    for batch_size in [8, 32, 128, 512]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        start_time = time.time()
        for batch in loader:
            pass  # Just iterate, measuring collation overhead
        total_time = time.time() - start_time

        batches = len(loader)
        time_per_batch = (total_time / batches) * 1000  # Convert to ms

        print(
            f"  Batch size {batch_size:3d}: {time_per_batch:.2f}ms per batch ({batches}"
            f" batches total)"
        )

    print("\n💡 Collation Insights:")
    print("• Larger batches take longer to collate (more np.stack operations)")
    print("• But fewer large batches are more efficient than many small ones")
    print("• Optimal: Balance between batch size and iteration overhead")


if __name__ == "__main__":
    analyze_dataloader_performance()
    analyze_memory_usage()
    analyze_collation_overhead()

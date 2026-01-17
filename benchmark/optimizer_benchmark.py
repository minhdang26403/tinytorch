import numpy as np

from tinytorch.optim import SGD, Adam, AdamW
from tinytorch.tensor import Tensor


def analyze_optimizer_memory_usage():
    """📊 Analyze memory usage of different optimizers."""
    print("📊 Analyzing Optimizer Memory Usage...")

    # Create test parameters of different sizes
    param_sizes = [1000, 10000, 100000]  # 1K, 10K, 100K parameters

    print("Optimizer Memory Analysis (per parameter tensor):")
    print("=" * 60)
    print(f"{'Size':<10} {'SGD':<10} {'Adam':<10} {'AdamW':<10} {'Ratio':<10}")
    print("-" * 60)

    for size in param_sizes:
        # Create parameter
        param = Tensor(np.random.randn(size), requires_grad=True)
        param.grad = Tensor(np.random.randn(size))

        # SGD memory (parameter + momentum buffer)
        sgd = SGD([param], momentum=0.9)
        sgd.step()  # Initialize buffers
        sgd_memory = size * 2  # param + momentum buffer

        # Adam memory (parameter + 2 moment buffers)
        param_adam = Tensor(np.random.randn(size), requires_grad=True)
        param_adam.grad = Tensor(np.random.randn(size))
        adam = Adam([param_adam])
        adam.step()  # Initialize buffers
        adam_memory = size * 3  # param + m_buffer + v_buffer

        # AdamW memory (same as Adam)
        adamw_memory = adam_memory

        # Memory ratio (Adam/SGD)
        ratio = adam_memory / sgd_memory

        print(
            f"{size:<10} {sgd_memory:<10} {adam_memory:<10} {adamw_memory:<10} "
            f"{ratio:.1f}x"
        )

    print("\n💡 Key Insights:")
    print("- SGD: 2× parameter memory (momentum buffer)")
    print("- Adam/AdamW: 3× parameter memory (two moment buffers)")
    print("- Memory scales linearly with model size")
    print("- Trade-off: More memory for better convergence")


def analyze_optimizer_convergence_behavior():
    """📊 Analyze convergence behavior of different optimizers."""
    print("📊 Analyzing Optimizer Convergence Behavior...")

    # Simulate optimization of a quadratic function: f(x) = 0.5 * x^2
    # Optimal solution: x* = 0, gradient = x

    def quadratic_loss(x):
        """Simple quadratic function for optimization testing."""
        return 0.5 * (x**2).sum()

    def compute_gradient(x):
        """Gradient of quadratic function: df/dx = x."""
        return x.copy()

    # Starting point
    x_start = np.array([5.0, -3.0, 2.0])  # Far from optimum [0, 0, 0]

    # Test different optimizers
    optimizers_to_test = [
        ("SGD", SGD, {"lr": 0.1}),
        ("SGD+Momentum", SGD, {"lr": 0.1, "momentum": 0.9}),
        ("Adam", Adam, {"lr": 0.1}),
        ("AdamW", AdamW, {"lr": 0.1, "weight_decay": 0.01}),
    ]

    print("Convergence Analysis (quadratic function f(x) = 0.5 * x²):")
    print("=" * 70)
    print(
        f"{'Optimizer':<15} {'Step 0':<12} {'Step 5':<12} {'Step 10':<12}"
        f"{'Final Loss':<12}"
    )
    print("-" * 70)

    for name, optimizer_class, kwargs in optimizers_to_test:
        # Reset parameter
        param = Tensor(x_start.copy(), requires_grad=True)
        optimizer = optimizer_class([param], **kwargs)

        losses = []

        # Run optimization for 10 steps
        for step in range(11):
            # Compute loss and gradient
            loss = quadratic_loss(param.data)
            param.grad = Tensor(compute_gradient(param.data))

            losses.append(loss)

            # Update parameters
            if step < 10:  # Don't update after last evaluation
                optimizer.step()
                optimizer.zero_grad()

        # Format results
        step0 = f"{losses[0]:.6f}"
        step5 = f"{losses[5]:.6f}"
        step10 = f"{losses[10]:.6f}"
        final = f"{losses[10]:.6f}"

        print(f"{name:<15} {step0:<12} {step5:<12} {step10:<12} {final:<12}")

    print("\n💡 Key Insights:")
    print("- SGD: Steady progress but can be slow")
    print("- SGD+Momentum: Faster convergence, less oscillation")
    print("- Adam: Adaptive rates help with different parameter scales")
    print("- AdamW: Similar to Adam with regularization effects")


if __name__ == "__main__":
    analyze_optimizer_memory_usage()
    analyze_optimizer_convergence_behavior()

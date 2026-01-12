import numpy as np

from tinytorch.nn import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss, log_softmax
from tinytorch.tensor import Tensor

# Constants for numerical stability
EPSILON = 1e-7  # Small value to prevent log(0) and numerical instability


def test_unit_log_softmax():
    # Test basic functionality
    x = Tensor([[1.0, 2.0, 3.0], [0.1, 0.2, 0.9]])
    result = log_softmax(x, dim=-1)

    # Verify shape preservation
    assert result.shape == x.shape, (
        f"Shape mismatch: expected {x.shape}, got {result.shape}"
    )

    # Verify log-softmax properties: exp(log_softmax) should sum to 1
    softmax_result = np.exp(result.data)
    row_sums = np.sum(softmax_result, axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"Softmax doesn't sum to 1: {row_sums}"
    )

    # Test numerical stability with large values
    large_x = Tensor([[100.0, 101.0, 102.0]])
    large_result = log_softmax(large_x, dim=-1)
    assert not np.any(np.isnan(large_result.data)), (
        "NaN values in result with large inputs"
    )
    assert not np.any(np.isinf(large_result.data)), (
        "Inf values in result with large inputs"
    )


def test_unit_mse_loss():
    loss_fn = MSELoss()

    # Test perfect predictions (loss should be 0)
    predictions = Tensor([1.0, 2.0, 3.0])
    targets = Tensor([1.0, 2.0, 3.0])
    perfect_loss = loss_fn.forward(predictions, targets)
    assert np.allclose(perfect_loss.data, 0.0, atol=EPSILON), (
        f"Perfect predictions should have 0 loss, got {perfect_loss.data}"
    )

    # Test known case
    predictions = Tensor([1.0, 2.0, 3.0])
    targets = Tensor([1.5, 2.5, 2.8])
    loss = loss_fn.forward(predictions, targets)

    expected_loss = (0.25 + 0.25 + 0.04) / 3
    assert np.allclose(loss.data, expected_loss, atol=1e-6), (
        f"Expected {expected_loss}, got {loss.data}"
    )

    # Test that loss is always non-negative
    random_pred = Tensor(np.random.randn(10))
    random_target = Tensor(np.random.randn(10))
    random_loss = loss_fn.forward(random_pred, random_target)
    assert random_loss.data >= 0, (
        f"MSE loss should be non-negative, got {random_loss.data}"
    )


def test_unit_cross_entropy_loss():
    loss_fn = CrossEntropyLoss()

    # Test perfect predictions (should have very low loss)
    perfect_logits = Tensor(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]]
    )  # Very confident predictions
    targets = Tensor([0, 1])  # Matches the confident predictions
    perfect_loss = loss_fn.forward(perfect_logits, targets)
    assert perfect_loss.data < 0.01, (
        f"Perfect predictions should have very low loss, got {perfect_loss.data}"
    )

    # Test uniform predictions (should have loss ≈ log(num_classes))
    uniform_logits = Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])  # Equal probabilities
    uniform_targets = Tensor([0, 1])
    uniform_loss = loss_fn.forward(uniform_logits, uniform_targets)
    expected_uniform_loss = np.log(3)  # log(3) ≈ 1.099 for 3 classes
    assert np.allclose(uniform_loss.data, expected_uniform_loss, atol=0.1), (
        f"Uniform predictions should have loss ≈ log(3) = {expected_uniform_loss:.3f}, "
        f"got {uniform_loss.data:.3f}"
    )

    # Test that wrong confident predictions have high loss
    wrong_logits = Tensor(
        [[10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]]
    )  # Confident but wrong
    wrong_targets = Tensor([1, 1])  # Opposite of confident predictions
    wrong_loss = loss_fn.forward(wrong_logits, wrong_targets)
    assert wrong_loss.data > 5.0, (
        f"Wrong confident predictions should have high loss, got {wrong_loss.data}"
    )

    # Test numerical stability with large logits
    large_logits = Tensor([[100.0, 50.0, 25.0]])
    large_targets = Tensor([0])
    large_loss = loss_fn.forward(large_logits, large_targets)
    assert not np.isnan(large_loss.data), "Loss should not be NaN with large logits"
    assert not np.isinf(large_loss.data), (
        "Loss should not be infinite with large logits"
    )


def test_unit_binary_cross_entropy_loss():
    loss_fn = BinaryCrossEntropyLoss()

    # Test perfect predictions
    perfect_predictions = Tensor([0.9999, 0.0001, 0.9999, 0.0001])
    targets = Tensor([1.0, 0.0, 1.0, 0.0])
    perfect_loss = loss_fn.forward(perfect_predictions, targets)
    assert perfect_loss.data < 0.01, (
        f"Perfect predictions should have very low loss, got {perfect_loss.data}"
    )

    # Test worst predictions
    worst_predictions = Tensor([0.0001, 0.9999, 0.0001, 0.9999])
    worst_targets = Tensor([1.0, 0.0, 1.0, 0.0])
    worst_loss = loss_fn.forward(worst_predictions, worst_targets)
    assert worst_loss.data > 5.0, (
        f"Worst predictions should have high loss, got {worst_loss.data}"
    )

    # Test uniform predictions (probability = 0.5)
    uniform_predictions = Tensor([0.5, 0.5, 0.5, 0.5])
    uniform_targets = Tensor([1.0, 0.0, 1.0, 0.0])
    uniform_loss = loss_fn.forward(uniform_predictions, uniform_targets)
    expected_uniform = -np.log(0.5)  # Should be about 0.693
    assert np.allclose(uniform_loss.data, expected_uniform, atol=0.01), (
        f"Uniform predictions should have loss ≈ {expected_uniform:.3f}, got "
        f"{uniform_loss.data:.3f}"
    )

    # Test numerical stability at boundaries
    boundary_predictions = Tensor([0.0, 1.0, 0.0, 1.0])
    boundary_targets = Tensor([0.0, 1.0, 1.0, 0.0])
    boundary_loss = loss_fn.forward(boundary_predictions, boundary_targets)
    assert not np.isnan(boundary_loss.data), "Loss should not be NaN at boundaries"
    assert not np.isinf(boundary_loss.data), "Loss should not be infinite at boundaries"


def test_loss():
    # Run all unit tests
    print("Running unit tests...")
    test_unit_log_softmax()
    test_unit_mse_loss()
    test_unit_cross_entropy_loss()
    test_unit_binary_cross_entropy_loss()

    print("\nRunning integration scenarios...")

    # Simulate a complete prediction -> loss computation pipeline

    # 1. MSE for regression (house price prediction)
    house_predictions = Tensor(
        [250.0, 180.0, 320.0, 400.0]
    )  # Predicted prices in thousands
    house_actual = Tensor([245.0, 190.0, 310.0, 420.0])  # Actual prices
    mse_loss = MSELoss()
    house_loss = mse_loss.forward(house_predictions, house_actual)
    assert house_loss.data > 0, "House price loss should be positive"
    assert house_loss.data < 1000, "House price loss should be reasonable"

    # 2. CrossEntropy for classification (image recognition)
    image_logits = Tensor(
        [[2.1, 0.5, 0.3], [0.2, 2.8, 0.1], [0.4, 0.3, 2.2]]
    )  # 3 images, 3 classes
    image_labels = Tensor([0, 1, 2])  # Correct class for each image
    ce_loss = CrossEntropyLoss()
    image_loss = ce_loss.forward(image_logits, image_labels)
    assert image_loss.data > 0, "Image classification loss should be positive"
    assert image_loss.data < 5.0, "Image classification loss should be reasonable"

    # 3. BCE for binary classification (spam detection)
    spam_probabilities = Tensor([0.85, 0.12, 0.78, 0.23, 0.91])
    spam_labels = Tensor([1.0, 0.0, 1.0, 0.0, 1.0])  # True spam labels
    bce_loss = BinaryCrossEntropyLoss()
    spam_loss = bce_loss.forward(spam_probabilities, spam_labels)
    assert spam_loss.data > 0, "Spam detection loss should be positive"
    assert spam_loss.data < 5.0, "Spam detection loss should be reasonable"

    # 4. Test numerical stability with extreme values
    extreme_logits = Tensor([[100.0, -100.0, 0.0]])
    extreme_targets = Tensor([0])
    extreme_loss = ce_loss.forward(extreme_logits, extreme_targets)
    assert not np.isnan(extreme_loss.data), "Loss should handle extreme values"
    assert not np.isinf(extreme_loss.data), "Loss should not be infinite"


# Run comprehensive module test
if __name__ == "__main__":
    test_loss()

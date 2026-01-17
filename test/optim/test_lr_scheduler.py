from tinytorch.optim.lr_scheduler import CosineSchedule


def test_unit_cosine_schedule():
    """🔬 Test CosineSchedule implementation."""
    print("🔬 Unit Test: CosineSchedule...")

    # Test basic schedule
    schedule = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)

    # Test start, middle, and end
    lr_start = schedule.get_lr(0)
    lr_middle = schedule.get_lr(50)
    lr_end = schedule.get_lr(100)

    print(f"Learning rate at epoch 0: {lr_start:.4f}")
    print(f"Learning rate at epoch 50: {lr_middle:.4f}")
    print(f"Learning rate at epoch 100: {lr_end:.4f}")

    # Validate behavior
    assert abs(lr_start - 0.1) < 1e-6, f"Expected 0.1 at start, got {lr_start}"
    assert abs(lr_end - 0.01) < 1e-6, f"Expected 0.01 at end, got {lr_end}"
    assert 0.01 < lr_middle < 0.1, (
        f"Middle LR should be between min and max, got {lr_middle}"
    )

    # Test monotonic decrease in first half
    lr_quarter = schedule.get_lr(25)
    assert lr_quarter > lr_middle, "LR should decrease monotonically in first half"

    print("✅ CosineSchedule works correctly!")

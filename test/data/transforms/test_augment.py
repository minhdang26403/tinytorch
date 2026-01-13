import numpy as np

from tinytorch import Tensor
from tinytorch.data.transforms import Compose, RandomCrop, RandomHorizontalFlip


def test_unit_augmentation():
    print("\n Unit Test: Data Augmentation...")

    # Test 1: RandomHorizontalFlip
    print("  Testing RandomHorizontalFlip...")
    flip = RandomHorizontalFlip(p=1.0)  # Always flip for deterministic test

    img = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))  # 2x3 image
    flipped = flip(img)
    expected = Tensor(np.array([[3, 2, 1], [6, 5, 4]]))
    assert np.array_equal(flipped.data, expected.data), (
        f"Flip failed: {flipped} vs {expected}"
    )

    # Test never flip
    no_flip = RandomHorizontalFlip(p=0.0)
    unchanged = no_flip(img)
    assert np.array_equal(unchanged.data, img.data), "p=0 should never flip"

    # Test 2: RandomCrop shape preservation
    print("  Testing RandomCrop...")
    crop = RandomCrop(32, padding=4)

    # Test with (C, H, W) format (CIFAR-10 style)
    img_chw = Tensor(np.random.randn(3, 32, 32))
    cropped = crop(img_chw)
    assert cropped.shape == (3, 32, 32), f"CHW crop shape wrong: {cropped.shape}"

    # Test with (H, W) format
    img_hw = Tensor(np.random.randn(28, 28))
    crop_hw = RandomCrop(28, padding=4)
    cropped_hw = crop_hw(img_hw)
    assert cropped_hw.shape == (28, 28), f"HW crop shape wrong: {cropped_hw.shape}"

    # Test 3: Compose pipeline
    print("  Testing Compose...")
    transforms = Compose([RandomHorizontalFlip(p=0.5), RandomCrop(32, padding=4)])

    img = Tensor(np.random.randn(3, 32, 32))
    augmented = transforms(img)
    assert augmented.shape == (3, 32, 32), (
        f"Compose output shape wrong: {augmented.shape}"
    )

    # Test 4: Transforms work with Tensor
    print("  Testing Tensor compatibility...")
    tensor_img = Tensor(np.random.randn(3, 32, 32))

    flip_result = RandomHorizontalFlip(p=1.0)(tensor_img)
    assert isinstance(flip_result, Tensor), (
        "Flip should return Tensor when given Tensor"
    )

    crop_result = RandomCrop(32, padding=4)(tensor_img)
    assert isinstance(crop_result, Tensor), (
        "Crop should return Tensor when given Tensor"
    )

    # Test 5: Randomness verification
    print("  Testing randomness...")
    flip_random = RandomHorizontalFlip(p=0.5)

    # Run many times and check we get both outcomes
    flips = 0
    no_flips = 0
    test_img = Tensor(np.array([[1, 2]]))

    for _ in range(100):
        result = flip_random(test_img)
        if np.array_equal(result.data, np.array([[2, 1]])):
            flips += 1
        else:
            no_flips += 1

    # With p=0.5, we should get roughly 50/50 (allow for randomness)
    assert flips > 20 and no_flips > 20, (
        f"Flip randomness seems broken: {flips} flips, {no_flips} no-flips"
    )

    print("✅ Data Augmentation works correctly!")

import numpy as np

from tinytorch import Tensor


class RandomCrop:
    """
    Randomly crop image after padding.

    This is the standard augmentation for CIFAR-10:
    1. Pad image by `padding` pixels on each side
    2. Randomly crop back to original size

    This simulates small translations in the image, forcing the model
    to recognize objects regardless of their exact position.

    Args:
        size: Output crop size (int for square, or tuple (H, W))
        padding: Pixels to pad on each side before cropping (default: 4)
    """

    def __init__(self, size, padding=4):
        """
        Initialize RandomCrop.

        APPROACH:
        1. Convert size to tuple if it's an int (for square crops)
        2. Store size and padding as instance variables

        EXAMPLE:
        >>> crop = RandomCrop(32, padding=4)  # CIFAR-10 standard
        >>> # Pads to 40x40, then crops back to 32x32
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply random crop after padding.

        Args:
            x: Input image with shape (C, H, W) or (H, W) or (H, W, C)
               Assumes spatial dimensions are H, W

        Returns:
            Cropped image with target size

        EXAMPLE:
        >>> crop = RandomCrop(32, padding=4)
        >>> img = Tensor(np.random.randn(3, 32, 32))  # CIFAR-10 format (C, H, W)
        >>> out = crop(img)
        >>> print(out.shape)  # (3, 32, 32)
        """
        target_h, target_w = self.size

        # Determine image format and dimensions
        if len(x.shape) == 2:
            # (H, W) format
            h, w = x.shape
            padded = np.pad(x.data, self.padding, mode="constant", constant_values=0)

            max_top = h + 2 * self.padding - target_h
            max_left = w + 2 * self.padding - target_h

            # Random crop position
            top = np.random.randint(0, max_top + 1)
            left = np.random.randint(0, max_left + 1)

            cropped = padded[top : top + target_h, left : left + target_w]

        elif len(x.shape) == 3:
            if x.shape[0] <= 4:  # Likely (C, H, W) format
                _, h, w = x.shape
                # Pad only spatial dimensions
                padded = np.pad(
                    x.data,
                    (
                        (0, 0),
                        (self.padding, self.padding),
                        (self.padding, self.padding),
                    ),
                    mode="constant",
                    constant_values=0,
                )

                max_top = h + 2 * self.padding - target_h
                max_left = w + 2 * self.padding - target_h

                # Random crop position
                top = np.random.randint(0, max_top + 1)
                left = np.random.randint(0, max_left + 1)

                cropped = padded[:, top : top + target_h, left : left + target_w]
            else:  # Likely (H, W, C) format
                h, w, _ = x.shape
                padded = np.pad(
                    x.data,
                    (
                        (self.padding, self.padding),
                        (self.padding, self.padding),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=0,
                )

                max_top = h + 2 * self.padding - target_h
                max_left = w + 2 * self.padding - target_h

                top = np.random.randint(0, max_top + 1)
                left = np.random.randint(0, max_left + 1)

                cropped = padded[top : top + target_h, left : left + target_w, :]
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {x.shape}")

        return Tensor(cropped)


class RandomHorizontalFlip:
    """
    Randomly flip images horizontally with given probability.

    A simple but effective augmentation for most image datasets.
    Flipping is appropriate when horizontal orientation doesn't change class
    (cats, dogs, cars - not digits or text!).

    Args:
        p: Probability of flipping (default: 0.5)
    """

    def __init__(self, p=0.5):
        """
        Initialize RandomHorizontalFlip.

        EXAMPLE:
        >>> flip = RandomHorizontalFlip(p=0.5)  # 50% chance to flip
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Probability must be between 0 and 1, got {p}")
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply random horizontal flip to input.

        Args:
            x: Input tensor with shape (..., H, W) or (..., H, W, C)
               Flips along the width dimension

        Returns:
            Flipped or unchanged tensor (same shape as input)

        EXAMPLE:
        >>> flip = RandomHorizontalFlip(0.5)
        >>> img = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))  # 2x3 image
        >>> # 50% chance output is [[3, 2, 1], [6, 5, 4]]
        """
        if np.random.random() < self.p:
            if len(x.shape) == 2:
                # (H, W) format
                flip_axis = -1
            elif len(x.shape) == 3:
                if x.shape[-1] <= 4:
                    # Likely (H, W, C) format
                    flip_axis = -2
                else:
                    # Likely (C, H, W) format
                    flip_axis = -1
            else:
                raise ValueError(f"Expected 2D or 3D input, got shape {x.shape}")

            return Tensor(np.flip(x.data, axis=flip_axis))
        return x


class Compose:
    """
    Compose multiple transforms into a pipeline.

    Applies transforms in sequence, passing output of each
    as input to the next.

    Args:
        transforms: List of transform callables
    """

    def __init__(self, transforms):
        """
        Initialize Compose with list of transforms.

        EXAMPLE:
        >>> transforms = Compose([
        ...     RandomHorizontalFlip(0.5),
        ...     RandomCrop(32, padding=4)
        ... ])
        """
        self.transforms = transforms

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply all transforms in sequence.
        """
        for transform in self.transforms:
            x = transform(x)
        return x

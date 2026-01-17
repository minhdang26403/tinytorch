from tinytorch.nn.utils import im2col
from tinytorch.tensor import Tensor


class MaxPool2d:
    """
    2D Max Pooling layer for spatial dimension reduction.

    Applies maximum operation over spatial windows, preserving
    the strongest activations while reducing computational load.

    Args:
        kernel_size: Size of pooling window (int or tuple)
        stride: Stride of pooling operation (default: same as kernel_size)
        padding: Zero-padding added to input (default: 0)
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        """
        Initialize MaxPool2d layer.
        """

        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Default stride equals kernel_size (non-overlapping)
        if stride is None:
            self.stride = self.kernel_size[0]
        else:
            self.stride = stride

        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through MaxPool2d layer.
        """
        # Input validation and shape extraction
        if len(x.shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch, channels, height, width), got {x.shape}"
            )

        n, c_in, h_in, w_in = x.shape
        kernel_h, kernel_w = self.kernel_size

        # Perform im2col operation
        # (kernel_h * kernel_w * c_in, n * h_out * w_out)
        cols = im2col(x.data, kernel_h, kernel_w, self.stride, self.padding)

        # Reshape columns to (c_in, kernel_h * kernel_w, n, h_out, w_out)
        h_out = (h_in + 2 * self.padding - kernel_h) // self.stride + 1
        w_out = (w_in + 2 * self.padding - kernel_w) // self.stride + 1
        cols = cols.reshape(c_in, -1, n, h_out, w_out)

        # Find the maximum value in each window
        out = cols.max(axis=1).transpose(1, 0, 2, 3)
        return Tensor(out)

    def parameters(self):
        """Return empty list (pooling has no parameters)."""
        return []

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)


class AvgPool2d:
    """
    2D Average Pooling layer for spatial dimension reduction.

    Applies average operation over spatial windows, smoothing
    features while reducing computational load.

    Args:
        kernel_size: Size of pooling window (int or tuple)
        stride: Stride of pooling operation (default: same as kernel_size)
        padding: Zero-padding added to input (default: 0)
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        """
        Initialize AvgPool2d layer.
        """
        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Default stride equals kernel_size (non-overlapping)
        if stride is None:
            self.stride = self.kernel_size[0]
        else:
            self.stride = stride

        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through AvgPool2d layer.
        """
        # Input validation and shape extraction
        if len(x.shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch, channels, height, width), got {x.shape}"
            )

        n, c_in, h_in, w_in = x.shape
        kernel_h, kernel_w = self.kernel_size

        # Perform im2col operation
        # (kernel_h * kernel_w * c_in, n * h_out * w_out)
        cols = im2col(x.data, kernel_h, kernel_w, self.stride, self.padding)

        # Reshape columns to (c_in, kernel_h * kernel_w, n, h_out, w_out)
        h_out = (h_in + 2 * self.padding - kernel_h) // self.stride + 1
        w_out = (w_in + 2 * self.padding - kernel_w) // self.stride + 1
        cols = cols.reshape(c_in, -1, n, h_out, w_out)

        # Find mean value in each window
        out = cols.mean(axis=1).transpose(1, 0, 2, 3)

        return Tensor(out)

    def parameters(self):
        """Return empty list (pooling has no parameters)."""
        return []

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)

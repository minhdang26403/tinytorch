import numpy as np

from tinytorch.nn.utils import im2col
from tinytorch.tensor import Tensor


class Conv2d:
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        """
        Initialize Conv2d layer with proper weight initialization.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        # He initialization for ReLU networks
        kernel_h, kernel_w = self.kernel_size
        fan_in = in_channels * kernel_h * kernel_w
        std = np.sqrt(2.0 / fan_in)

        # Weight shape: (out_channels, in_channels, kernel_h, kernel_w)
        self.weight = Tensor(
            np.random.normal(0, std, (out_channels, in_channels, kernel_h, kernel_w))
        )

        # Bias initialization
        if bias:
            self.bias = Tensor(np.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through Conv2d layer.
        """

        if len(x.shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch, channels, height, width), got {x.shape}"
            )

        n, _, h_in, w_in = x.shape
        c_out, _, kernel_h, kernel_w = self.weight.shape

        # (c_out, c_in * kernel_h * kernel_w)
        rows = self.weight.data.reshape(self.out_channels, -1)
        # (c_in * kernel_h * kernel_w, n * h_out * w_out)
        cols = im2col(x.data, kernel_h, kernel_w, self.stride, self.padding)

        # (c_out, n * h_out * w_out)
        output = rows @ cols
        h_out = (h_in + 2 * self.padding - kernel_h) // self.stride + 1
        w_out = (w_in + 2 * self.padding - kernel_w) // self.stride + 1
        output = output.reshape(n, c_out, h_out, w_out)

        if self.bias is not None:
            # Broadcast bias across the batch and spatial dimensions
            output = output + self.bias.data.reshape(1, c_out, 1, 1)

        return Tensor(output)

    def parameters(self):
        """Return trainable parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)

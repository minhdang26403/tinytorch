import numpy as np

from tinytorch.nn.utils import im2col
from tinytorch.tensor import Tensor


class Conv2d:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups=1,
        stride=1,
        padding=0,
        bias=True,
    ):
        """
        Initialize Conv2d layer with proper weight initialization.

        Convolution may split input and output channels into groups,
        performing separate convolutions for each group. This is used in
        architectures like ResNeXt and depthwise separable convolutions.
        """
        # Validate groups before assigning
        assert in_channels % groups == 0, (
            f"in_channels ({in_channels}) must be divisible by groups ({groups})"
        )
        assert out_channels % groups == 0, (
            f"out_channels ({out_channels}) must be divisible by groups ({groups})"
        )

        self.groups = groups
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
        fan_in = (in_channels // groups) * kernel_h * kernel_w
        std = np.sqrt(2.0 / fan_in)

        # Weight shape: (out_channels, in_channels, kernel_h, kernel_w)
        self.weight = Tensor(
            np.random.normal(
                0, std, (out_channels, in_channels // groups, kernel_h, kernel_w)
            )
        )

        # Bias initialization
        if bias:
            self.bias = Tensor(np.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch, channels, height, width), got {x.shape}"
            )

        x_split = np.split(x.data, self.groups, axis=1)
        w_splits = np.split(self.weight.data, self.groups, axis=0)

        output_chunks = []
        for i in range(self.groups):
            chunk_out = self._single_conv2d(x_split[i], w_splits[i])
            output_chunks.append(chunk_out)

        output = np.concatenate(output_chunks, axis=1)
        if self.bias is not None:
            output = output + self.bias.data.reshape(1, -1, 1, 1)

        return Tensor(output)

    def _single_conv2d(self, x_data: np.ndarray, w_data: np.ndarray) -> np.ndarray:
        n, _, h_in, w_in = x_data.shape
        c_out, _, kernel_h, kernel_w = w_data.shape

        # (c_out, c_in * kernel_h * kernel_w)
        w_rows = w_data.reshape(c_out, -1)
        # (c_in * kernel_h * kernel_w, n * h_out * w_out)
        x_cols = im2col(x_data, kernel_h, kernel_w, self.stride, self.padding)

        # (c_out, n * h_out * w_out)
        output = w_rows @ x_cols
        h_out = (h_in + 2 * self.padding - kernel_h) // self.stride + 1
        w_out = (w_in + 2 * self.padding - kernel_w) // self.stride + 1
        output = output.reshape(c_out, n, h_out, w_out).transpose(1, 0, 2, 3)

        return output

    def parameters(self):
        """Return trainable parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)

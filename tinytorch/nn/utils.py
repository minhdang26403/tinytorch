import numpy as np

from tinytorch.tensor import Tensor


def clip_grad_norm(parameters: list[Tensor], max_norm: float = 1.0) -> float:
    if not parameters:
        return 0.0

    # Collect all gradients
    sum_squared_grad = 0.0
    for param in parameters:
        if param.grad is None:
            continue

        grad = param.grad
        grad_data = grad.data if isinstance(grad, Tensor) else grad
        sum_squared_grad += np.sum(grad_data**2)

    # Compute global norm
    total_norm = np.sqrt(sum_squared_grad)

    # Clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        for param in parameters:
            if param.grad is None:
                continue

            if isinstance(param.grad, Tensor):
                param.grad.data = param.grad.data * clip_coef
            else:
                param.grad = param.grad * clip_coef

    return total_norm


def im2col(
    x: np.ndarray, kernel_h: int, kernel_w: int, stride: int, padding: int
) -> np.ndarray:
    """
    Convert input tensor to column format for convolution.

    Args:
        x: Input tensor of shape (n, c_in, h_in, w_in)
        kernel_h: Height of the kernel
        kernel_w: Width of the kernel
        stride: Stride of the convolution
        padding: Padding of the convolution

    Returns:
        cols: Column tensor of shape (c_in * kernel_h * kernel_w, n * h_out * w_out)
    """

    n, c_in, h_in, w_in = x.shape
    h_out = (h_in + 2 * padding - kernel_h) // stride + 1
    w_out = (w_in + 2 * padding - kernel_w) // stride + 1

    cols = np.zeros((c_in * kernel_h * kernel_w, n * h_out * w_out))

    if padding > 0:
        padded_x = np.pad(
            x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
        )
    else:
        padded_x = x

    col_idx = 0
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * stride
            w_start = j * stride
            patch = padded_x[
                :, :, h_start : h_start + kernel_h, w_start : w_start + kernel_w
            ]
            cols[:, col_idx : col_idx + n] = patch.reshape(n, -1).transpose()
            col_idx += n

    return cols

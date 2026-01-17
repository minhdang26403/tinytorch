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

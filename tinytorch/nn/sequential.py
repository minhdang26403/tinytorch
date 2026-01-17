from tinytorch.tensor import Tensor

from .module import Module


class Sequential(Module):
    """
    Container that chains layers together sequentially.
    """

    def __init__(self, *layers: Module):
        """
        Initialize with layers to chain together.
        """
        # Accept both Sequential(layer1, layer2) and Sequential([layer1, layer2])
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers = tuple(layers[0])
        else:
            self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all layers sequentially.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Tensor]:
        """
        Collect all parameters from all layers.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self) -> str:
        layer_reprs = ", ".join(repr(layer) for layer in self.layers)
        return f"Sequential({layer_reprs})"

from ..tensor import Tensor


class MSELoss:
    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        pass

    def backward(self, grad: Tensor) -> Tensor:
        pass


class CrossEntropyLoss:
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        pass

    def backward(self, grad: Tensor) -> Tensor:
        pass


class BinaryCrossEntropyLoss:
    def forward(self, predicitons: Tensor, target: Tensor) -> Tensor:
        pass

    def backward(self, grad: Tensor) -> Tensor:
        pass

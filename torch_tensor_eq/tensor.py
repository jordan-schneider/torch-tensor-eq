import torch


class Tensor(torch.Tensor):
    """torch Tensor, except `__eq__` returns a bool, matching the python convention."""

    def __eq__(self, other: torch.Tensor) -> bool:
        """Returns true if all elements of other are equal to the corresponding elements of self."""
        return torch.all(torch.eq(self, other))

    def __ne__(self, other: torch.Tensor) -> bool:
        """Returns false if any element of other doesn't match a corresponding element of self. """
        return not self.__eq__(other)


def eq(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """Returns an elementwise equality between two Tensors."""
    return torch.eq(tensor1, tensor2)

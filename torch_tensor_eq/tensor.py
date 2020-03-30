import hashlib
import json

import torch


class Tensor(torch.Tensor):
    """torch Tensor, except `__eq__` returns a bool, matching the python convention."""

    def __hash__(self, *args, **kwargs):
        # Deterministic hash by value
        return hashlib.md5(json.dumps(self))

    def __eq__(self, other) -> bool:  # type: ignore
        """Returns true if all elements of other are equal to the corresponding elements of self."""
        return isinstance(other, torch.Tensor) and bool(
            torch.all(torch.eq(self, other)).item()
        )

    def __ne__(self, other) -> bool:  # type: ignore
        """Returns false if any element of other doesn't match a corresponding element of self. """
        return not self.__eq__(other)

    def __lt__(self, other) -> bool:  # type: ignore
        "A vector is not boolean less than another vector unless both are length 1."
        if isinstance(other, Tensor) and self.shape == (1,) and other.shape == 1:
            return self.item() < other.item()
        return False

    def __le__(self, other) -> bool:  # type: ignore
        return self == other

    def __gt__(self, other) -> bool:  # type: ignore
        if isinstance(other, Tensor) and self.shape == (1,) and other.shape == 1:
            return self.item() > other.item()
        return False

    def __ge__(self, other) -> bool:  # type: ignore
        return self == other


def eq(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """Returns an elementwise equality between two Tensors."""
    out: torch.Tensor = torch.eq(tensor1, tensor2)
    return Tensor(out)

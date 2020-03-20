# torch-tensor-eq
Provides a subclass of torch.Tensor whose equal function returns a bool.

For some reason torch returns arrays of bools when you use their built in `__eq__`
methods. This makes it impossible to compare containers of tensors, since containers (sensibly)
assume they can rely on their memeber's `__eq__` methods to return bool.

This package provides a subclass of torch.Tensor which overrides the default torch `__eq__` to
return a single bool, and provides an `eq()` method which provides the elementwise functionality.
The `eq()` method is totally superfluous, since `torch.eq()` also provides this functionality.

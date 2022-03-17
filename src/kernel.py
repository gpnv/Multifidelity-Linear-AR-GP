import torch
from typing import Optional
from gpytorch.lazy import InterpolatedLazyTensor,  RootLazyTensor, CatLazyTensor
from gpytorch.priors import Prior
from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.kernels import Kernel
from gpytorch.lazy import InterpolatedLazyTensor, RootLazyTensor


class LowFidelityIndexKernel(Kernel):
    def __init__(
        self,
        num_tasks: int,
        rank: Optional[int] = 1,
        prior: Optional[Prior] = None,
        **kwargs,
    ):
        if rank > num_tasks:
            raise RuntimeError("Cannot create a task covariance matrix larger than the number of tasks")
        super().__init__(**kwargs)

        self.register_parameter(
            name="rho", parameter=torch.nn.Parameter(torch.randn(*self.batch_shape, 1))
        )

        if prior is not None:
            if not isinstance(prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(prior).__name__)
            self.register_prior("IndexKernelPrior", prior, lambda m: m._eval_covar_matrix())

    def _eval_covar_matrix(self):
        cf = torch.stack([torch.tensor([1]), self.rho])
        return cf @ cf.transpose(-1, -2)

    @property
    def covar_matrix(self):
        res = RootLazyTensor(CatLazyTensor([torch.tensor([1]), self.rho]))
        return res

    def forward(self, i1, i2, **params):
        i1, i2 = i1.long(), i2.long()
        covar_matrix = self._eval_covar_matrix()
        batch_shape = _mul_broadcast_shape(i1.shape[:-2], i2.shape[:-2], self.batch_shape)
        res = InterpolatedLazyTensor(
            base_lazy_tensor=covar_matrix,
            left_interp_indices=i1.expand(batch_shape + i1.shape[-2:]),
            right_interp_indices=i2.expand(batch_shape + i2.shape[-2:]),
        )
        return res

class HighFidelityIndexKernel(Kernel):
    def __init__(
        self,
        num_tasks: int,
        rank: Optional[int] = 1,
        **kwargs,
    ):
        if rank > num_tasks:
            raise RuntimeError("Cannot create a task covariance matrix larger than the number of tasks")
        super().__init__(**kwargs)
        self.covar_factor = torch.tensor([[0],[1]])

    def _eval_covar_matrix(self):
        cf = self.covar_factor
        return cf @ cf.transpose(-1, -2)

    @property
    def covar_matrix(self):
        res = self.covar_factor @ self.covar_factor.transpose(-1, -2)
        return res

    def forward(self, i1, i2, **params):
        
        i1, i2 = i1.long(), i2.long()
        covar_matrix = self._eval_covar_matrix()
        batch_shape = _mul_broadcast_shape(i1.shape[:-2], i2.shape[:-2], self.batch_shape)
        res = InterpolatedLazyTensor(
            base_lazy_tensor=covar_matrix,
            left_interp_indices=i1.expand(batch_shape + i1.shape[-2:]),
            right_interp_indices=i2.expand(batch_shape + i2.shape[-2:]),
        )
        return res
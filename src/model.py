from kernel import LowFidelityIndexKernel, HighFidelityIndexKernel
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
import tqdm

class LinearARGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, HF_kernel, LF_kernel, rank=1, num_tasks=2, feature_extractor = None):
        super(LinearARGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.HF_kernel = HF_kernel
        self.LF_kernel = LF_kernel

        self.HF_task_kernel = HighFidelityIndexKernel()
        self.LF_task_kernel = LowFidelityIndexKernel()

        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x, i):
        if not (self.feature_extractor is None):
            x = self.feature_extractor(x)
            x = self.scale_to_bounds(x)

        mean_x = self.mean_module(x)
        
        # hf
        covar_hf_x = self.HF_kernel(x)
        covar_hf_i = self.HF_task_kernel(i)

        # lf
        covar_lf_x = self.LF_kernel(x)
        covar_lf_i = self.LF_task_kernel(i)

        # Combine
        covar1 = covar_hf_x.mul(covar_hf_i)
        covar2 = covar_lf_x.mul(covar_lf_i)
        covar = covar1+covar2

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

class LinearARModel():
    def __init__(self, full_train_x, full_train_y, full_train_i, name=None, lr=0.1, epoch=500, use_ARD = False) -> None:
        self.full_train_x = full_train_x
        self.full_train_i = full_train_i
        self.full_train_y = full_train_y
        self.lr = lr
        self.use_ARD = use_ARD
        self.epoch = epoch
        self.in_size = full_train_x.shape[-1]
        self.out_size = full_train_y.shape[-1]
        self.name = name if name else self.__class__.__name__

    def train(self):

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

        optimizer = torch.optim.Adam([
                {'params': self.model.parameters()}
            ],
            lr=self.lr
        )

        MF_lml = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        self.likelihood.train()
        self.model.train()

        iterator = tqdm.tqdm(range(1, self.epoch + 1))
        for _ in iterator:
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = self.model(self.full_train_x, self.full_train_i)
            loss = -MF_lml(output, self.full_train_y)
            loss.backward(retain_graph=True)
            optimizer.step()
            iterator.set_postfix({'loss': loss.item()})

    def build(self):
        # Kernel
        d = self.in_size if self.use_ARD else 1
        HF_kernel = ScaleKernel(RBFKernel(ard_num_dims=d))
        LF_kernel = ScaleKernel(RBFKernel(ard_num_dims=d))

        # Model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = LinearARGPModel((self.full_train_x, self.full_train_i), self.full_train_y, 
            self.likelihood, HF_kernel=HF_kernel, LF_kernel=LF_kernel)

        self.train()

    def predict(self, x, i):
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(x,i))
        return pred.mean, pred.variance

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

clamp_with_grad = ClampWithGrad.apply 

class LogitDistributionLoss(nn.Module):
    def __init__(self, distribution='laplace'):
        super().__init__()          
        self.distribution = distribution

    def forward(self, mean, logvar, target):
        if self.distribution == 'laplace':
            dist = torch.distributions.Laplace(mean, logvar.exp())
        else:
            dist = torch.distributions.Normal(mean, logvar.exp())
        loss = -dist.log_prob(target).mean() 

        return loss
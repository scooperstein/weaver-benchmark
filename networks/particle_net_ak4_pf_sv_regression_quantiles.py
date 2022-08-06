import numpy as np
import math
import torch
from torch import Tensor
from nn.model.ParticleNet import ParticleNetTagger


def get_model(data_config, **kwargs):

    conv_params = [
        (16, (192, 160, 128)),
        (14, (192, 160, 128)),
        (12, (192, 160, 128)),
        ]

    fc_params = [
        (192, 0.1),
        (160, 0.1),
        (128, 0.1),
        (96, 0.1),
        (64, 0.1)
    ]

    input_dims = 48;
    use_fusion = True

    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    num_classes = 0
    num_targets = len(data_config.target_value)
    model = ParticleNetTagger(pf_features_dims, 
                              sv_features_dims, 
                              num_classes,
                              num_targets,
                              conv_params, 
                              fc_params,
                              input_dims=input_dims,
                              use_fusion=use_fusion,
                              use_fts_bn=kwargs.get('use_fts_bn', False),
                              use_counts=kwargs.get('use_counts', True),
                              pf_input_dropout=kwargs.get('pf_input_dropout', None),
                              sv_input_dropout=kwargs.get('sv_input_dropout', None),
                              for_inference=kwargs.get('for_inference', False)
                              )

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['output'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output':{0:'N'}}},
        }

    return model, model_info

class LogCoshLoss(torch.nn.L1Loss):

    __constants__ = ['reduction','quantiles']

    def __init__(self, reduction: str = 'mean') -> None:
        super(LogCoshLoss, self).__init__(None, None, reduction)
        self.quantiles = quantiles;

    def forward(self, input: Tensor, target: Tensor, quantiles: Tensor) -> Tensor:

        ## account for the presence of quantiles
        x = input - target;
        for idx, q in enumerate(self.quantiles):
            if q <= 0 and idx == 0:
                loss = x[idx] + torch.nn.functional.softplus(-2. * x[idx]) - math.log(2);
            elif q <= 0 and idx != 0:
                loss += x[idx] + torch.nn.functional.softplus(-2. * x[idx]) - math.log(2);
            elif q > 0 and idx == 0:
                loss  = q*x[idx]*torch.ge(x[idx],0)
                loss += (q-1)*(x[idx])*torch.less(x[idx],0);
            elif q > 0 and idx != 0:
                loss += q*x[idx]*torch.ge(x[idx],0)
                loss += (q-1)*(x[idx])*torch.less(x[idx],0);
                
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

def get_loss(data_config, **kwargs):
    quantiles = data_config.target_quantile;
    return LogCoshLoss(reduction='mean',quantiles=quantiles);

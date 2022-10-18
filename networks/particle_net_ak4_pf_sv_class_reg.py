import numpy as np
import math
import torch
from torch import Tensor
from nn.model.ParticleNet import ParticleNetTagger

def get_model(data_config, **kwargs):

    ## input numer of point features to EdgeConvBlock
    point_features = 48;
    ## convoluational layers in EdgeConvBlock and kNN
    conv_params = [
        (16, (224, 192, 128)),
        (14, (224, 192, 160)),
        (12, (224, 192, 160))
        ]
    ## use fusion layer for edge-conv block
    use_fusion = True
    ## fully connected output layers
    fc_params = [
        (192, 0.1),
        (160, 0.1),
        (128, 0.1),
        (96, 0.1),
        (64, 0.1)
    ]

    ## classes and features
    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    num_classes = len(data_config.label_value);
    num_targets = len(data_config.target_value)

    model = ParticleNetTagger(pf_features_dims=pf_features_dims, 
                              sv_features_dims=sv_features_dims, 
                              num_classes=num_classes,
                              num_targets=num_targets,
                              conv_params=conv_params, 
                              fc_params=fc_params,
                              input_dims=point_features, 
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


class CrossEntropyLogCoshLoss(torch.nn.L1Loss):
    __constants__ = ['reduction','nclass','ntarget','loss_lambda']

    def __init__(self, reduction: str = 'mean', nclass: int = 1, ntarget: int = 1, loss_lambda: float = 1.) -> None:
        super(CrossEntropyLogCoshLoss, self).__init__(None, None, reduction)
        self.nclass = nclass;
        self.ntarget = ntarget;
        self.loss_lambda = loss_lambda

    def forward(self, input: Tensor, y_cat: Tensor, y_reg: Tensor) -> Tensor:

        ## regression term
        input_reg = input[:,self.nclass:self.nclass+self.ntarget].squeeze();
        y_reg     = y_reg.squeeze();
        loss_reg  = (input_reg-y_reg)+torch.nn.functional.softplus(-2.*(input_reg-y_reg))-math.log(2);
        ## classification term
        input_cat = input[:,:self.nclass].squeeze();
        y_cat     = y_cat.squeeze().long();
        loss_cat  = torch.nn.functional.cross_entropy(input_cat,y_cat,reduction=self.reduction);
                
        ## final loss and pooling over batcc
        if self.reduction == 'none':            
            return loss_cat+self.loss_lambda*loss_reg, loss_cat, loss_reg*self.loss_lambda;
        elif self.reduction == 'mean':
            return loss_cat+self.loss_lambda*loss_reg.mean(), loss_cat, loss_reg.mean()*self.loss_lambda;
        elif self.reduction == 'sum':
            return loss_cat+self.loss_lambda*loss_reg.sum(), loss_cat, loss_reg.sum()*self.loss_lambda;


def get_loss(data_config, **kwargs):
    nclass  = len(data_config.label_value);
    ntarget = len(data_config.target_value);
    return CrossEntropyLogCoshLoss(reduction=kwargs.get('reduction','mean'),loss_lambda=kwargs.get('loss_lambda',1),nclass=nclass,ntarget=ntarget);

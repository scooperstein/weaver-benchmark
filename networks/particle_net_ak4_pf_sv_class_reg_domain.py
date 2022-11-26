import numpy as np
import math
import torch
from torch import Tensor
from nn.model.ParticleNet import ParticleNetLostTrkTagger

def get_model(data_config, **kwargs):

    ## input numer of point features to EdgeConvBlock
    point_features = 48;
    ## convoluational layers in EdgeConvBlock and kNN
    conv_params = [
        (16, (256, 208, 176)),
        (14, (256, 208, 176)),
        (12, (256, 208, 176))
        ]
    ## use fusion layer for edge-conv block
    use_fusion = True
    ## fully connected output layers
    fc_params = [
        (224, 0.1),
        (192, 0.1),
        (160, 0.1),
        (128, 0.1),
        (96,  0.1),
        (64,  0.1)
    ]
    ## fully connected output layers
    fc_domain_params = [
        (224, 0.1),
        (192, 0.1),
        (160, 0.1),
        (128, 0.1),
        (96,  0.1),
        (64,  0.1)
    ]

    ## classes and features
    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    lt_features_dims = len(data_config.input_dicts['lt_features'])
    num_classes = len(data_config.label_value);
    num_targets = len(data_config.target_value)
    num_domains = len(data_config.domain_value)

    model = ParticleNetLostTrkTagger(pf_features_dims=pf_features_dims, 
                                     sv_features_dims=sv_features_dims, 
                                     lt_features_dims=lt_features_dims, 
                                     num_classes=num_classes,
                                     num_targets=num_targets,
                                     num_domains=num_domains,
                                     conv_params=conv_params, 
                                     fc_params=fc_params,
                                     fc_domain_params=fc_domain_params,
                                     input_dims=point_features, 
                                     use_fusion=use_fusion,
                                     use_fts_bn=kwargs.get('use_fts_bn', False),
                                     use_counts=kwargs.get('use_counts', True),
                                     pf_input_dropout=kwargs.get('pf_input_dropout', None),
                                     sv_input_dropout=kwargs.get('sv_input_dropout', None),
                                     sv_input_dropout=kwargs.get('lt_input_dropout', None),
                                     for_inference=kwargs.get('for_inference', False)
                                 )

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['output'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output':{0:'N'}}},
        }

    return model, model_info


class CrossEntropyLogCoshLossDomain(torch.nn.L1Loss):
    __constants__ = ['reduction','nclass','ntarget','ndomain','loss_lambda','loss_gamma','quantiles','loss_kappa']

    def __init__(self, reduction: str = 'mean', nclass: int = 1, ntarget: int = 1, ndomain: int = 1, 
                 loss_lambda: float = 1., loss_gamma: float = 1., quantiles: list = [], loss_kappa: float 1.) -> None:
        super(CrossEntropyLogCoshLoss, self).__init__(None, None, reduction)
        self.nclass = nclass;
        self.ntarget = ntarget;
        self.ndomain = ndomain;
        self.loss_lambda = loss_lambda;
        self.loss_gamma = loss_gamma;
        self.loss_kappa = loss_kappa;
        self.quantiles = quantiles;

    def forward(self, input: Tensor, y_cat: Tensor, y_reg: Tensor, y_domain: Tensor) -> Tensor:

        ## classification term
        input_cat = input[:,:self.nclass].squeeze();
        y_cat     = y_cat.squeeze().long();
        loss_cat  = torch.nn.functional.cross_entropy(input_cat,y_cat,reduction=self.reduction);

        ## regression terms
        input_reg  = input[:,self.nclass:self.nclass+self.ntarget].squeeze();
        y_reg      = y_reg.squeeze();
        x_reg      = input_reg-y_reg;

        loss_mean  = torch.zeros(size=(0,1));
        loss_quant = torch.zeros(size=(0,1));

        for idx,q in enumerate(self.quantiles):
            if q <= 0 and loss_mean.nelement()==0:
                loss_mean = (x_reg[:,idx])+torch.nn.functional.softplus(-2.*(x_reg[:,idx]))-math.log(2);
            elif q <= 0:
                loss_mean += (x_reg[:,idx])+torch.nn.functional.softplus(-2.*(x_reg[:,idx]))-math.log(2);
            if q > 0 and loss_quant.nelement()==0:
                loss_quant  = q*x_reg[:,idx]*torch.ge(x_reg[:,idx],0)
                loss_quant += (q-1)*(x_reg[:,idx])*torch.less(x_reg[:,idx],0);
            elif q > 0:
                loss_quant += q*x_reg[:,idx]*torch.ge(x_reg[:,idx],0)
                loss_quant += (q-1)*(x_reg[:,idx])*torch.less(x_reg[:,idx],0);                

        loss_reg = self.loss_lambda*loss_mean+self.loss_gamma*loss_quant;

        ## domain terms
        input_domain  = input[:,self.nclass+self.ntarget:self.nclass+self.ntarget+self.ndomain].squeeze();
        y_domain      = y_domain.squeeze();
        loss_domain   = self.loss_kappa*torch.nn.functional.cross_entropy(input_domain,y_domain,reduction=self.reduction);

        ## over a batch
        if self.reduction == 'none':            
            return loss_cat+loss_reg+loss_domain, loss_cat, loss_reg, loss_domain;
        elif self.reduction == 'mean':
            return loss_cat+loss_reg.mean()+loss_domain, loss_cat, loss_reg.mean(), loss_domain;
        elif self.reduction == 'sum':
            return loss_cat+loss_reg.sum()+loss_domain, loss_cat, loss_reg.sum(), loss_domain;

def get_loss(data_config, **kwargs):

    nclass  = len(data_config.label_value);
    ntarget = len(data_config.target_value);
    ndomain = len(data_config.domain_value);
    quantiles = data_config.target_quantile;

    return 
    CrossEntropyLogCoshLossDomain(
        reduction=kwargs.get('reduction','mean'),
        loss_lambda=kwargs.get('loss_lambda',1),
        loss_gamma=kwargs.get('loss_gamma',1),
        loss_kappa=kwargs.get('loss_kappa',1),
        nclass=nclass,
        ntarget=ntarget,
        ndomain=ndomain,
        quantiles=quantiles
    );
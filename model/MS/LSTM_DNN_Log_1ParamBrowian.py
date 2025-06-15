import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base.LSTM_DNN_1ParamBrowian import BaseRTF



class LLT(nn.Module):
    '''Learnable Logarithm Transformer'''
    def __init__(self,phi=10,_delta=1e-7):
        super().__init__()
        self.phi = nn.Parameter(torch.FloatTensor([phi]))
        self.delta = _delta
    def forward(self,x):
        return torch.log(F.relu(x+self.phi)+self.delta)

class MSRTF(BaseRTF):
    '''Multi-Stage model for RTF dataset'''
    def __init__(self,phi=10,_delta=1e-7,**Base_kwargs):
        super().__init__(**Base_kwargs)
        self.fix_point = Base_kwargs['args'].MS_fix_point
        self.hi_transformer = LLT(phi,_delta)

    def transform_deg_hi(self,hi):
        if self.fix_point > 0:
            return self.hi_transformer(hi[self.fix_point-1:])

    def mfe_loss(self,x,UUT, reduction = 'mean'):
        x = self.transform_deg_hi(x)
        return super().mfe_loss(x,UUT, reduction)
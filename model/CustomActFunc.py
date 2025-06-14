import torch
import torch.nn as nn
import torch.nn.functional as F

def select_activate(activate_type,**kw_activate):
    if activate_type == 'SigmoidExpBias':
        return SigmoidExpBias(**kw_activate)
    elif activate_type == 'SigmoidLinear':
        return SigmoidLinear(**kw_activate)
    elif activate_type == 'SigmoidLeakyReLU':
        return SigmoidLeakeyReLU(**kw_activate)
    elif activate_type == 'SigmoidELU':
        return SigmoidELU(**kw_activate)
    else:
        return nn.Sigmoid()

class SigmoidExpBias(nn.Module):
    def __init__(self,bias=0):
        super().__init__()
        self.bias = nn.Parameter(torch.FloatTensor([bias]))
    
    def forward(self,x):
        return F.sigmoid(torch.exp(x)+self.bias)



class SigmoidLinear(nn.Module):
    def __init__(self,weight=None,bias=0):
        super().__init__()
        if weight is None:
            self.weight = nn.Parameter(torch.FloatTensor(size=(1,)))
            nn.init.uniform_(self.weight)
        else:
            self.weight = nn.Parameter(torch.FloatTensor([weight]))
        self.bias = nn.Parameter(torch.FloatTensor([bias]))
    
    def forward(self,x):
        return F.sigmoid(self.weight*x+self.bias)



class SigmoidLeakeyReLU(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self,x):
        return F.sigmoid(F.leaky_relu(x,self.alpha))



class SigmoidELU(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self,x):
        return F.sigmoid(F.elu(x,self.alpha))
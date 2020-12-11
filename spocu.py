import torch.nn as nn
import torch
import numpy as np

class SPOCU(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def h_function(self,value):
        h_ = value.detach().numpy()
        h_ = h_.clip(0)
        h_ = pow(h_,3)*(pow(h_,5)-(2*pow(h_,4))+2)
        h_ = torch.from_numpy(h_)

        return h_
    
    def h2_function(self, value):
        if value > 0:
            return pow(value,3)*(pow(value,5)-(2*pow(value,4))+2)
        else:
            return 0

    def forward(self, input):
        out = self.alpha*self.h_function((input/self.gamma)+self.beta) - self.alpha*self.h2_function(self.beta)
        return out

if __name__ == '__main__':
  alpha = 3.0937
  beta = 0.6653
  gamma = 4.437
  
  spocu = SPOCU(alpha, beta, gamma)
  
  x = torch.rand((10,10))
  print(spocu(x))

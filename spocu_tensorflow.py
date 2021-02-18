import tensorflow as tf
from tensorflow import keras
import numpy as np

class SPOCU(keras.layers.Layer):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def h_function(self,value):
        h_ = value
        #h_ = value.numpy()
        clip_val_max = tf.math.reduce_max(tf.math.reduce_max(h_))
        h_ = tf.clip_by_value(h_,0,clip_val_max)
        h_ = pow(h_,3)*(pow(h_,5)-(2*pow(h_,4))+2)
        #h_ = torch.from_numpy(h_)

        return h_
    
    def h2_function(self, value):
        if value > 0:
            return pow(value,3)*(pow(value,5)-(2*pow(value,4))+2)
        else:
            return 0

    def call(self, input):
        out = self.alpha*self.h_function((input/self.gamma)+self.beta) - self.alpha*self.h2_function(self.beta)
        return out

if __name__ == '__main__':
  alpha = 3.0937
  beta = 0.6653
  gamma = 4.437
  
  spocu = SPOCU(alpha, beta, gamma)
  
  X = tf.Variable(tf.random.normal([10, 10], stddev=5, mean=4) )
  print(type(X))
  print(spocu(X))

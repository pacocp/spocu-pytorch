import torch
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
#tf.get_logger().setLevel('ERROR')


from spocu.spocu_pytorch import SPOCU as SPOCUpytorch
from spocu.spocu_tensorflow import SPOCU as SPOCUtensorflow 

if __name__ == '__main__':
  alpha = 3.0937
  beta = 0.6653
  gamma = 4.437
  

  for index in range(100):
    print(index)
    spocu_pytorch1 = SPOCUpytorch(alpha, beta, gamma)
    spocu_tensorflow1 = SPOCUtensorflow(alpha, beta, gamma)
    
    np.random.seed(index)
    x_np = np.random.normal(size=(10,10))
    #print(x_np)
    x_pytorch = torch.Tensor(x_np)
    result_pytorch = spocu_pytorch1(x_pytorch).detach().numpy()
    x_tensorflow = tf.convert_to_tensor(x_np)
    result_tensorflow = spocu_tensorflow1(x_tensorflow).numpy()
    #assert result_pytorch == result_tensorflow
    #print(result_pytorch)
    #print(result_tensorflow)
    # need to use low precision atol due to 32 bit and 64 bit defaults in pytorch and tensorflow
    assert np.allclose(result_pytorch,result_tensorflow,atol=1e-05)

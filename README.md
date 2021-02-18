# “SPOCU”: scaled polynomial constant unit activation function.

Non-official Pytorch/Tensorflow implementation of the SPOCU activation function [1], for the case when
c=infinite.


## Pytorch

It can be included in your network given an alpha, beta and gamma value:

```
from spocu_pytorch import SPOCU

alpha = 3.0937
beta = 0.6653
gamma = 4.437

spocu = SPOCU(alpha, beta, gamma)

x = torch.rand((10,10))
print(spocu(x))

```



## Tensorflow


```
from spocu_tensorflow import SPOCU

alpha = 3.0937
beta = 0.6653
gamma = 4.437

spocu = SPOCU(alpha, beta, gamma)

  
X = tf.Variable(tf.random.normal([10, 10], stddev=5, mean=4) )
print(spocu(X))

```


## Tests

See [spocu_test](spocu_test.py) for equivalance of pytorch and tensorflow implementation.


## Acknowledgements

Thanks to the author of the Tensorflow version, [Atilla Ozgur](https://github.com/ati-ozgur).

# Bibliography

```
[1] Kiseľák, J., Lu, Y., Švihra, J. et al. “SPOCU”: scaled polynomial constant unit activation function. Neural Comput & Applic (2020). https://doi.org/10.1007/s00521-020-05182-1
```



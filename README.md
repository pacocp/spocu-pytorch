# “SPOCU”: scaled polynomial constant unit activation function.


Non-official Pytorch implementation of the SPOCU activation function [1], for the case when
c=infinite.

It can be included in your network given a alpha, beta and gamma value:

```
from spocu import SPOCU

alpha = 3.0937
beta = 0.6653
gamma = 4.437

spocu = SPOCU(alpha, beta, gamma)

x = torch.rand((10,10))
print(spocu(x))

```



# Bibliography

```
[1] Kiseľák, J., Lu, Y., Švihra, J. et al. “SPOCU”: scaled polynomial constant unit activation function. Neural Comput & Applic (2020). https://doi.org/10.1007/s00521-020-05182-1
```



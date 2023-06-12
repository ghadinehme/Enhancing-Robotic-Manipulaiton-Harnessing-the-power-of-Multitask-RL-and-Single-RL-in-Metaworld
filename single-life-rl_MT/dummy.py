import numpy as np 




array = np.concatenate([np.zeros((10,1)), np.ones((10,1))], axis = 1)

array = np.random.randn(10)

idx = np.array([1,2,5])


print(array[idx])
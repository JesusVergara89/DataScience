#L1 = list(range(10))

#print(L1)
#print(type(L1[0]),type(L1[1]),type(L1[2]))

#L2 = [str(item) for item in L1]

#print(L2) 

#####################################################

#L3 = [True, "2", 3.0, 4]

#print([type(item) for item in L3])

#####################################################
#import array
#L = list(range(10))
#L[2] = 'A'
#A = array.array('i', L)
#print(A)
#####################################################
#import numpy as np 

#L = [1.2,2,'a',4,5]

#L1 = np.array(L, dtype='int')

#print(L1)
#####################################################
#import numpy as np 

#L1 = [2,4,6]

#L2 = np.array([range(i,i+3) for i in L1])

#print(L2)
#####################################################
#import numpy as np

#L1 = np.zeros(10, dtype=int)

#print(L1)
#####################################################
#import numpy as np

#L1 = np.full((3,5), 3.14)

#print(L1)

#####################################################
#import numpy as np

#L1 = np.arange(0,20,2)

#print(L1)

#####################################################
#import numpy as np

#L1 = np.linspace(0,1,5)

#print(L1)

#####################################################
#import numpy as np

#L1 = np.random.random((3,3))

#print(L1)
#####################################################
#import numpy as np

#L1 = np.random.normal(0,1,(3,3))

#print(L1)
#####################################################
#import numpy as np

#L1 = np.random.randint(0,10,(3,3))

#print(L1)
#####################################################
#import numpy as np

#L1 = np.eye(3)

#print(L1)
#####################################################
#import numpy as np

#L1 = np.empty(3)

#print(L1)
#####################################################
#import numpy as np

#L1 = np.zeros(10, dtype=np.int16)

#print(L1)
#####################################################

#import numpy as np

#np.random.seed(0)

#x1 = np.random.randint(10, size=6) # One-dimensional array
#x2 = np.random.randint(10, size=(3,4)) # Two-dimensional array
#x3 = np.random.randint(10, size=(3,4,6)) # Three-dimensional array

#print(f'x1 = {x1}','\n\n',f'x2 = {x2}','\n\n',f'x3 = {x3}')

#print("x3 ndim: ", x3.ndim)
#print("x3 shape:", x3.shape)
#print("x3 size: ", x3.size)
#print("dtype:", x3.dtype)
#####################################################

#import numpy as np

#L1 = np.arange(3)

#L2 = [list(range(i,i+3)) for i in L1]

#L3 = np.array(L2)

#print((L3[2,-1]))
#####################################################
#import numpy as np

#L1 = np.arange(0,10)

#print(L1)

#L1[0] = 3.14

#print(L1)

#####################################################
#import numpy as np

#x = np.arange(10)

#print(x[:5])
#print(x[5:])
#print(x[4:7])
#print(x[::2])
#print(x[1::2])
#print(x[::-1])
#print(x[5::-2])

#####################################################

import numpy as np

np.random.seed(0)

x = np.random.randint(0,20,(4,5))

#print(x)

#print(x[:,1::2])

print(x[::-1,::-1])

#####################################################
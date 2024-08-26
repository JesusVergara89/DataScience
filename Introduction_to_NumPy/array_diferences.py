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

#import numpy as np

#np.random.seed(0)

#x = np.random.randint(0,20,(4,5))

#print(x)

#print(x[:,1::2])

#print(x[::-1,::-1])

#####################################################
"""
import numpy as np

np.random.seed(0)

x = np.random.randint(0,20,(4,5))

print(x)
print(x[:2,:2])

x_subs = (x[:2,:2])
x_subs[0,0] = 222

print(x)
"""
#####################################################
"""
import numpy as np

np.random.seed(0)

x = np.random.randint(0,20,(4,5))

x1 = (x[:2,:3]).copy()

x1[1,2] = 999

print(x)
print(x1)
"""

#####################################################
"""
import numpy as np

np.random.seed(0)

x = np.arange(1,4)

y = x.reshape((1,3))

print(y)

"""
"""
import numpy as np

np.random.seed(0)

x = np.arange(1,4)

y = x[np.newaxis, :]

print(y)
"""
"""
import numpy as np

np.random.seed(0)

x = np.arange(1,4)

y = x.reshape((3,1))

print(y)
"""
"""
import numpy as np

np.random.seed(0)

x = np.arange(1,4)

y = x[:, np.newaxis]

print(y)
"""
#####################################################
"""
import numpy as np

np.random.seed(0)

x = np.arange(1,5)

y = np.arange(6,10)

xy = np.concatenate([x,y])

print(xy)

z = np.arange(10,14)

xyz = np.concatenate([x,y,z])

print(xyz)
"""
"""
import numpy as np

np.random.seed(0)

x = np.random.randint(0,8,(2,3))

y = np.random.randint(0,8,(2,3))

xy = np.concatenate([x,y])

print(xy)
"""
"""
import numpy as np

np.random.seed(0)

x = np.random.randint(0,8,(2,3))

y = np.random.randint(0,8,(2,3))

xy = np.concatenate([x,y], axis=1)

print(xy)
"""
"""
import numpy as np

np.random.seed(0)

x = np.random.randint(1,10,(1,3))

y = np.random.randint(1,10,(3,3))

print(f'x: {x}', f'y: {y}', sep='\n\n')

xy = np.vstack([x,y])

print(xy)
"""
import numpy as np

np.random.seed(0)

x = np.random.randint(1,10,(2,3))

y = np.array([
    [99],
    [99]])

print(f'x: {x}', f'y: {y}', sep='\n\n')

print('\n')

xy = np.hstack([x,y])

print(xy)

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
"""
#####################################################
"""
import numpy as np

np.random.seed(0)

x = np.arange(10)

print(x)

x1,x2,x3 = np.split(x, [3,5])

print(f'x1: {x1}',f'x2: {x2}',f'x3: {x3}', sep='\n\n')
"""
"""
import numpy as np

np.random.seed(0)

x = np.arange(16).reshape((4,4))

print(x)

upper, lower = np.vsplit(x, [2])

#print('####################')

#print(f'upper: {upper}',f'lower: {lower}', sep='\n\n')

print('####################')

left, right = np.hsplit(x, [1])

print(f'left: {left}',f'right: {right}', sep='\n\n')
"""
#####################################################
"""
import numpy as np
import timeit

np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output

values = np.random.randint(1, 10, size=1000000)

#execution_time = timeit.timeit(lambda: compute_reciprocals(values), number=3)
#print(execution_time)

def compute_reciprocals_with_ufunc(values):
    return 1.0 / values

execution_time = timeit.timeit(lambda: compute_reciprocals_with_ufunc(values), number=100)
print(execution_time)
"""
#####################################################
"""
import numpy as np

np.random.seed(0)

x = np.random.randint(1, 10, size=10)

#y = np.random.randint(1, 10, size=10)

z = np.exp(x)

print(z, sep='|')
"""
#####################################################
"""
import numpy as np

np.random.seed(0)

x = np.arange(5)
y = np.arange(1,6)

print(x,y, sep='\n\n', end='\n\n')

print(x/y)
"""
#####################################################
"""
import numpy as np

np.random.seed(0)

x = np.arange(9).reshape((3,3))

print(x, end='\n\n')

y = np.power(x,2)

print(y)
"""
#####################################################
"""
import numpy as np

np.random.seed(0)

x = np.arange(4)

print(f'x = {x}',end='\n\n')
print(f'x + 5 = {x+5}',end='\n\n')
print(f'x - 5 = {x-5}',end='\n\n')
print(f'x * 2 = {x*2}',end='\n\n')
print(f'x / 2 = {x/2}',end='\n\n')
print(f'x // 2 = {x//2}',end='\n\n')
print(f'-x = {-x}',end='\n\n')
print(f'x ** 2 = {x**2}',end='\n\n')
print(f'x % 2 = {x%2}',end='\n\n')
"""
#####################################################
"""
import numpy as np

np.random.seed(0)

x = np.arange(6)

y = -x

print(y)

z = abs(y)

print(z) 

"""
"""
import numpy as np

np.random.seed(0)

real_parts = np.random.rand(5)  
imaginary_parts = np.random.rand(5) 

complex_numbers = real_parts + 1j * imaginary_parts

#print(complex_numbers)

y = abs(complex_numbers)

print(y)
"""
"""
#####################################################

import numpy as np

np.random.seed(0)

theta = np.linspace(0,np.pi,3)

#print(theta)

print("theta       = ", theta)
print("sin(theta)  = ", np.sin(theta))
print("cos(theta)  = ", np.cos(theta))
print("tan(theta)  = ", np.tan(theta))
"""
"""
#####################################################
import numpy as np

np.random.seed(0)

x = [-1,0,1]

print("x       = ",  x)
print("arcsin(theta)  = ", np.arcsin(x))
print("arccos(theta)  = ", np.arccos(x))
print("arctan(theta)  = ", np.arctan(x))
"""
#####################################################
"""
import numpy as np

np.random.seed(0)

x = [1,2,3]

print("x       = ",  x)
print("e^(x)  = ", np.exp(x))
print("2^(x)  = ", np.exp2(x))
print("2^(x)  = ", np.power(3,x))
"""
#####################################################
"""
import numpy as np

np.random.seed(0)

x = np.arange(5)

print(x)

y = np.empty(5)

np.multiply(x, 10, out=y)

print(y)
"""

#####################################################
"""
import numpy as np

np.random.seed(0)
x = np.arange(5)
y = np.zeros(10)

print(y)

np.power(2,x,out=y[::2])

print(y)
"""
#####################################################
"""
import numpy as np

np.random.seed(0)

x = np.arange(1,6)

print(x)

y = np.add.reduce(x)

print(y)
"""
"""
import numpy as np

np.random.seed(0)

x = np.arange(1,6)

print(x)

y = np.multiply.reduce(x)

print(y)
"""
"""
import numpy as np

np.random.seed(0)

x = np.arange(1,6)

print(x)

y = np.add.accumulate(x)
z = np.multiply.accumulate(x)
print(y)
print(z)
"""
#####################################################
"""
import numpy as np

np.random.seed(0)

x = np.arange(1,6)
print(x)
y = np.multiply.outer(x,x)
print(y)
"""
#####################################################
"""
import numpy as np
import timeit

np.random.seed(0)

x = np.random.random(50)

print(x)

y = sum(x)

z = np.sum(x)

execution_time1 = timeit.timeit(lambda: sum(x), number=1000000)
execution_time2 = timeit.timeit(lambda: np.sum(x), number=1000000)

print(y)
print(z)

print('Time python ', execution_time1)
print('Timer numpy ', execution_time2)
"""
"""
#####################################################
import numpy as np
import timeit

np.random.seed(0)

x = np.random.random(1000)

x1 = min(x)
x2 = max(x)

y1 = np.min(x)
y2 = np.max(x)

execution_timeX1 = timeit.timeit(lambda: min(x), number=3)
execution_timeX2 = timeit.timeit(lambda: max(x), number=3)

execution_timeY1 = timeit.timeit(lambda: np.min(x), number=3)
execution_timeY2 = timeit.timeit(lambda: np.max(x), number=3)


print('Time python ', execution_timeX1)
print('Time python ', execution_timeX2)
print('Timer numpy ', execution_timeY1)
print('Timer numpy ', execution_timeY2)
"""
#####################################################
"""
import numpy as np
import timeit

np.random.seed(0)

m = np.random.random((3,4))

#print(m)

#print(m.sum())

print('through columns = ', m.sum(axis=0))

print('through rows = ', m.sum(axis=1))

print('through columns = ', m.min(axis=0))

print('through rows = ', m.max(axis=1))
"""
#####################################################
"""
import numpy as np
import pandas as pd
import timeit

np.random.seed(0)

data = pd.read_csv('data_to_read/president_heights.csv')

#print(data)

heights = np.array(data['height(cm)'])

#print(heights)


#print(f'Mean heights:       {heights.mean()}')
#print(f'Standard deviation: {heights.std()}')
#print(f'Mininum heights:    {heights.min()}')
#print(f'Meximum heights:    {heights.max()}')


print(f'25th percentile:   {np.percentile(heights, 25)}')
print(f'Mediam:            {np.median(heights)}')
print(f'75th percentile:   {np.percentile(heights, 75)}')
"""
#####################################################
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(0)

data = pd.read_csv('data_to_read/president_heights.csv')

heights = np.array(data['height(cm)'])

plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number')

plt.show()
"""
#####################################################
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(0)

#a = np.array([0,1,2])
b = np.array([5,5,5])
#print(a)
#print(b)
#c = a+b
#print(c)
c = 5

a = b + c
print(a)
"""
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(0)

a = np.array([0,1,2])
m = np.ones((3,3))

print(a)

print(m)

print(m+a)
"""
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(0)

a = np.arange(3)
b = np.arange(3)[:, np.newaxis]

#print(a)
#print(b)

print(a+b)
"""
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(0)

m = np.ones((2,3))
a = np.arange(3)

print(m)
print(f'shape of m: {m.shape}')
print(a)
print(f'shape of a: {a.shape}')

print(m+a)
"""
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit
"""
"""
np.random.seed(0)

a = np.ones((3,2))
b = np.arange(3)
b = b[:,np.newaxis]

print(np.logaddexp(a,b))
"""
#####################################################
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(0)

x = np.random.random((10,3))

x_mean = x.mean(0)
print(x_mean)

x_centered = x - x_mean
print(x_centered)

print(x_centered.mean(0))
"""
#####################################################
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)

x = np.linspace(0,5,50)
y = np.linspace(0,5,50)[:,np.newaxis]


print(x.shape)
print(y.shape)

z = np.sin(x) ** 10 + np.cos(10 + y * x ) * np.cos(x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x, y)

ax.plot_surface(X, Y, z, cmap='viridis')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()
"""
#####################################################
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(0)

rainfall = pd.read_csv('data_to_read/Seattle2014.csv')['PRCP'].values

#print(rainfall)

inches = rainfall / 254  # 1/10mm -> inches

#print(inches.shape)

#plt.hist(inches, 40)

#plt.show()

#x = np.array([1,2,3,4,5,6,7,8,10])

#print((2*x)==(x**2))

#rng = np.random.RandomState(0)

#x = rng.randint(10, size=(3,4))
# < 
# >
#print(f'any is > 8    = {np.any(x > 8)}')
#print(f'any is < 0    = {np.any(x < 0)}')
#print(f'all are < 10  = {np.all(x < 10)}')
#print(f'all are == 6  = {np.all(x == 6)}')

#print(f'all are < 8 in each row  = {np.all(x < 8, axis=1)}')

#print(np.sum((inches > 0.5 ) & (inches < 1)))

print(f'Numbers of days without rain: {np.sum(inches == 0 )}')
print(f'Numbers of days with rain:    {np.sum(inches > 0 )}')
print(f'Numbers of days with more than 0.5inches of rain:    {np.sum(inches > 0.5 )}')
print(f'Rainy days with < 0.1 inches:    {np.sum(((inches > 0) & (inches < 0.2)))}')
"""
#####################################################
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(0)

rng = np.random.RandomState(0)

x = rng.randint(0,10,(3,4))

print(x)
print(x < 5 )

print(x[x < 5])
"""
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(0)

rainfall = pd.read_csv('data_to_read/Seattle2014.csv')['PRCP'].values

inches = rainfall / 254

#print(inches)
# < 
# >

rainy = ( inches > 0 )

#print(rainy)

summers_days = np.arange(365) - 172

#print(summers_days)

summer = ((np.arange(365) - 172 < 90 ) & (np.arange(365) - 172 > 0))

#print(summer)

print(f'Median precipitation on rainy days in 2014 (inches) = {np.median(inches[rainy])}')
print(f'Median precipitation on summer days in 2014 (inches) = {np.median(inches[summer])}')
print(f'Mex precipitation on summer days in 2014 (inches) = {np.max(inches[summer])}')
print(f'Median precip on non-summer rainy days (inches) = {np.median(inches[rainy & ~summer])}')
"""
#####################################################
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(0)

rand = np.random.RandomState(0)

x = rand.randint(100, size=10)

X = np.arange(12).reshape((3, 4))

# < 
# >

ind = np.array([[3,7],
                [4,5]])

#print(x[ind])

row = np.array([0,1,2])
col = np.array([2,1,3])

#print(X[row[:,np.newaxis], col])

y = row[:, np.newaxis] * col

print(y)
"""
#####################################################
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(0)

X = np.arange(12).reshape((3, 4))

#print(X[2,[2,0,1]])

#print(X[1:,[2,0,1]])

row = np.array([0,1,2])
col = np.array([2,1,3])

mask = np.array([1,0,1,0], dtype=bool)

#print(mask)

print(X[row[:, np.newaxis], mask])
"""
#####################################################
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(0)

#X = np.arange(12).reshape((3, 4))

mean = [0,0]
cov = [[1,2],
       [2,5]]
X = np.random.multivariate_normal(mean, cov, 100)

#plt.scatter(X[:,0],X[:,1])
#plt.show()

indices = np.random.choice(X.shape[0],20,replace=False)

#print(indices)

selection = X[indices]

#print(selection)
#print(selection.shape)

plt.scatter(X[:,0],X[:,1], alpha=0.3)
plt.scatter(selection[:, 0], selection[:, 1],facecolor='red',s=200)
plt.show()
"""
#####################################################
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(0)

x = np.arange(10)

y = { f'X[{i}]' : int(i) for i in x }

print(y)

i = np.array([2,1,8,4])

print(i)

x[i] = 99

print(x)
"""
#####################################################
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(0)

x = np.zeros(10)
print(x)

i = [2, 3, 3, 4, 4, 4]

y = np.add.at(x,i,1)

print(y)

"""
#####################################################
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(42)

x = np.random.randn(100)

bins = np.linspace(-5,5,20)

plt.hist(x, bins, histtype='step')

plt.show()
"""
#####################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set_theme()
import timeit

np.random.seed(0)

"""
def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

x = np.array([2,1,4,3,5])

compute_time_python = timeit.timeit(lambda: selection_sort(x), number=100)
compute_time_numpy = timeit.timeit(lambda: np.sort(x), number=100)

print(compute_time_python)
print(compute_time_numpy)
"""

# < 
# >
"""
def bogosort(x):
    while np.any(x[:-1] > x[1:]):
        np.random.shuffle(x)
    return x
x = np.array([2,1,4,3,5])
print(bogosort(x))
"""

#x = np.array([2,1,4,3,5])

#y = np.sort(x)

#print(y)

#i = np.argsort(x)

#print(i)

#print(x[i])

#rand = np.random.RandomState(42)

#x = rand.randint(0,10,(4,6))

#print(x)

#y = np.sort(x, axis =1)

#print(y)
"""
x = np.array([7,2,3,1,6,5,4])

y = np.partition(x,3)
z = np.partition(x,1)

print(y)
print(z)
"""
"""
y = np.partition(x,1, axis=0)
z = np.partition(x,1, axis=1)

print(y)
print(z)
"""
# < 
# >

x = np.random.randint(0,10,(10,2))

#print(x)

#plt.scatter(x[:,0],x[:,1], s=100)
#plt.show()

x1 = [x[:,np.newaxis,:]]
#print(x1)

x2 = [x[np.newaxis,:,:]]
print(x2)
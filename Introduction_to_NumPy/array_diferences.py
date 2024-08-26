#L1 = list(range(10))

#print(L1)
#print(type(L1[0]),type(L1[1]),type(L1[2]))

#L2 = [str(item) for item in L1]

#print(L2) 

#L3 = [True, "2", 3.0, 4]

#print([type(item) for item in L3])


#import array
#L = list(range(10))
#L[2] = 'A'
#A = array.array('i', L)
#print(A)

#import numpy as np 

#L = [1.2,2,'a',4,5]

#L1 = np.array(L, dtype='int')

#print(L1)

#import numpy as np 

#L1 = [2,4,6]

#L2 = np.array([range(i,i+3) for i in L1])

#print(L2)

#import numpy as np

#L1 = np.zeros(10, dtype=int)

#print(L1)

#import numpy as np

#L1 = np.full((3,5), 3.14)

#print(L1)


#import numpy as np

#L1 = np.arange(0,20,2)

#print(L1)


#import numpy as np

#L1 = np.linspace(0,1,5)

#print(L1)


#import numpy as np

#L1 = np.random.random((3,3))

#print(L1)

import numpy as np

L1 = np.random.normal(0,1,(3,3))

print(L1)
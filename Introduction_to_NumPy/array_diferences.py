#L1 = list(range(10))

#print(L1)
#print(type(L1[0]),type(L1[1]),type(L1[2]))

#L2 = [str(item) for item in L1]

#print(L2) 

#L3 = [True, "2", 3.0, 4]

#print([type(item) for item in L3])


import array

L = list(range(10))
L[2] = 'A'
A = array.array('i', L)
print(A)


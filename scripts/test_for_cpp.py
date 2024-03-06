import numpy as np

def hat_operator(a):
    res = np.zeros((3,3))
    res[0,1] = -a[2] 
    res[0,2] = a[1]
    res[1,0] = a[2]  
    res[1,2] = -a[0]
    res[2,0] = -a[1] 
    res[2,1] = a[0]
    return res

a = np.array([1,0,0])
b = np.array([0,1,0])

res = hat_operator(a)
print('res = ')
print(res)
print(res @ b)
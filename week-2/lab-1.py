import numpy as np

a = np.array([1,2,3,4])
print(f"a             : {a}")

b = -a 
print(f"b = -a        : {b}")

b = np.sum(a) 
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2      : {b}")

def my_dot(a, b): 
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 5, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")

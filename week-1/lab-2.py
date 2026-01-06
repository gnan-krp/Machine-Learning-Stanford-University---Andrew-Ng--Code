import numpy as np

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730])

def compute_cost(x_train, y_train, w, b): 
    m = x_train.shape[0] 
    cost_sum = 0
    
    for i in range(m): 
        f_wb = w * x_train[i] + b   
        cost = (f_wb - y_train[i]) ** 2  
        cost_sum = cost_sum + cost  
    
    total_cost = (1 / (2 * m)) * cost_sum  
    return total_cost

w = 100
b = 50
total_cost = compute_cost(x_train, y_train, w, b)
print("Total Cost =", total_cost)

import numpy as np
x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0,600.0])
print("x_train = ",x_train)
print("y_train = ",y_train)
print("x_train.shape:",x_train.shape)
m = x_train.shape[0]
print("Number of training examples is: ", {m})
print("y_train.shape:",y_train.shape)
j = y_train.shape[0]
print("Number of training examples is: ", {j})
for i in range(2):
    x_i = x_train[i]
    y_i = y_train[i]
    print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

w = 100
b = 50
x_i = 0.09
cost_i = w*x_i + b
print(f"Cost Function = {cost_i} Thousand Dollars")

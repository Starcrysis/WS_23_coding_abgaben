import numpy as np

y = np.array([3, 3/4, 1, 0, -9/4])
y_predict = np.array([3, 2, 1, 0, -1])
mean_y = np.mean(y)

TSS = np.sum([(y[i]-y_predict[i])**2 for i in range(len(y))])
RSS = np.sum([(elem - mean_y)**2 for elem in y])



# 3d

x = np.array([-2, -1, 0, 1, 2])

b0 = 0.5 
b1 = -9/8

new_y = [b0 + b1*x_i for x_i in x]
s2 = np.sum([(new_y - y_i)**2 for y_i in y])/len(y)

se1 = np.sqrt(s2*1/5)

se2 = np.sqrt(s2*1/10)


#2a
print(1/(1+np.exp(0.5)))
print(1/(1+np.exp(6-2.5-3.5)))
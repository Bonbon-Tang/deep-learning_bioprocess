import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c = np.loadtxt("normV_mean_std")
mean = c[0]
variance = c[1]

# input = np.loadtxt("1.1_160_0_noise.txt")
# input = input[::64,]
input = np.loadtxt("0.9_150_0_.txt")[0]
input = np.divide((input - mean), variance)
input = torch.Tensor(input)
#normalized input
real_output = np.loadtxt("0.9_150_0_.txt")
#real output
predict = np.zeros((301,3))

model8 = torch.load("model_16step")
model2 = torch.load("model_4step")
model_half = torch.load("model_1step")

for t in range (1,301):
    # if t % 64 == 0:
    #     a = int(t / 64)
    #     predict[t] = input.numpy()[a]
    # else:
    a = t
    n_8 = a // 16
    n_2 = (a - n_8 * 16) // 4
    n_half = (a - n_8 * 16 - n_2 * 4) // 1
    b = t // 64
    # initial = input[b]
    initial = input
    for i in range (n_8):
        result = model8(initial)
        initial = result
    for i in range (n_2):
        result = model2(initial)
        initial = result
    for i in range(n_half):
        result = model_half(initial)
        initial = result
    result = result.detach().numpy()
    predict[t] = result
predict[0] = input

predict_output = predict * variance + mean
print(predict_output)


fig,axes = plt.subplots(3,1)
ax1=axes[0]
ax2=axes[1]
ax3=axes[2]

ax1.plot(real_output[:,0],label = "real")
ax1.plot(predict_output[:,0],label = "predict")
ax1.legend()

ax2.plot(real_output[:,1],label = "real")
ax2.plot(predict_output[:,1],label = "predict")
ax2.legend()

ax3.plot(real_output[:,2],label = "real")
ax3.plot(predict_output[:,2],label = "predict")
ax3.legend()

plt.show()

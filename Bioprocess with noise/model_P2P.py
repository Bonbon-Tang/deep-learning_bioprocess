import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader,TensorDataset
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


step = 1

input_a = np.loadtxt("1.0_150_0_noise.txt")[0:-step,:]
output_a = np.loadtxt("1.0_150_0_noise.txt")[step:,:]
input_b = np.loadtxt("0.9_140_0_noise.txt")[0:-step,:]
output_b = np.loadtxt("0.9_140_0_noise.txt")[step:,:]
input_c = np.loadtxt("1.1_160_0.1_noise.txt")[0:-step,:]
output_c = np.loadtxt("1.1_160_0.1_noise.txt")[step:,:]

train_X = np.concatenate((input_a,input_b,input_c),axis=0)
train_Y = np.concatenate((output_a,output_b,output_c),axis=0)
val_X = np.loadtxt("0.95_145_0.1_noise.txt")[0:-step,:]
val_Y = np.loadtxt("0.95_145_0.1_noise.txt")[step:,:]

# a = np.mean(train_X,axis=0)
# b = np.std(train_X,axis=0)
# c = np.concatenate((a.reshape(1,3),b.reshape(1,3)),axis=0)
c = np.loadtxt("normV_mean_std")
a = c[0]
b = c[1]
m = torch.Tensor(a)
n = torch.Tensor(b)
# print(a)
# print(b)

train_X = np.divide((train_X-a),b)
train_Y = np.divide((train_Y-a),b)
# train_X = preprocessing.scale(train_X,axis=0)
# train_Y = preprocessing.scale(train_Y,axis=0)
train_X = torch.Tensor(train_X)
train_Y = torch.Tensor(train_Y)

val_X = np.divide((val_X-a),b)
val_X = torch.Tensor(val_X)
val_Y = np.divide((val_Y-a),b)
val_Y = torch.Tensor(val_Y)

model = nn.Sequential(nn.Linear(3,20),
                      nn.ReLU(),
                      nn.Linear(20,3),
                     )
for m in model.modules():
   if isinstance(m,(nn.Linear)):
        nn.init.kaiming_uniform_(m.weight)

model = torch.load("model_1step")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

def train(epochs):
    train_loss = []
    val_loss = []
    min_loss = 1e6
    for epoch in range(epochs):
            model.train()
            output = model(train_X)
            loss = criterion(output,train_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            model.eval()
            output_val = model(val_X)
            loss_dev = criterion(output_val,val_Y)
            if loss < min_loss:
                min_loss = loss
                torch.save(model, "model1")
            if epoch % 500 == 0:
                train_loss.append(loss.tolist())
                val_loss.append(loss_dev.tolist())
                print(f"epoch:{epoch},train_loss:{loss},val_loss:{loss_dev}")
    return train_loss,val_loss

train_loss,val_loss = train(2001)

model.eval()
input = np.loadtxt("1.1_160_0_.txt")[0:-step*4,:]

real_output = np.loadtxt("1.1_160_0_.txt")[step*4:,:]

input = np.divide((input-a),b)

input = torch.Tensor(input)

# model = torch.load("model_0.5s_final")

for i in range (4):
  input = model(input).cpu()
output = input.detach().numpy()
predict_output = np.multiply(output,b) + a

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

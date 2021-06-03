import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

step = 1

input_a = np.loadtxt("1.0_150_0_noise.txt")[0,:]
output_a = np.loadtxt("1.0_150_0_noise.txt")[1:301,:]
input_b = np.loadtxt("0.9_140_0_noise.txt")[0,:]
output_b = np.loadtxt("0.9_140_0_noise.txt")[1:301,:]
input_c = np.loadtxt("1.1_160_0.1_noise.txt")[0,:]
output_c = np.loadtxt("1.1_160_0.1_noise.txt")[1:301,:]


train_X = np.zeros((3,1,3))
train_X[0] = input_a
train_X[1] = input_b
train_X[2] = input_c
train_Y = np.zeros((3,300,3))
train_Y[0] = output_a
train_Y[1] = output_b
train_Y[2] = output_c
val_X = np.loadtxt("1.1_160_0_.txt")[0,:].reshape(1,1,3)
val_Y = np.loadtxt("1.1_160_0_.txt")[1:301,:].reshape(1,300,3)

c = np.loadtxt("normV_mean_std")
a = c[0]
b = c[1]

train_X = np.divide((train_X-a),b)
train_Y = np.divide((train_Y-a),b)
train_X = torch.Tensor(train_X)
train_Y = torch.Tensor(train_Y)
train_X,train_Y =train_X.to(device),train_Y.to(device)


val_X = np.divide((val_X-a),b)
val_Y = np.divide((val_Y-a),b)
val_X = torch.Tensor(val_X)
val_Y = torch.Tensor(val_Y)



class RNN(nn.Module):
    def __init__(self, input_size,hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.n_layers = n_layers

        self.gru = nn.GRU(input_size, hidden_size, n_layers,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input : torch.Tensor, predict_length = 300):
        hidden = None
        encoder_outputs, hidden = self.gru(input,hidden)
        decoder_input = encoder_outputs[:,-1,:].reshape(-1,1,self.hidden_size)
        decoder_input = self.out(decoder_input)
        decoder_output_set = torch.zeros([0]).to(device)
        decoder_output_set = torch.cat((decoder_output_set, decoder_input), dim=1)
        for _ in range (predict_length -1):
            one_output, hidden = self.gru (decoder_input,hidden)
            decoder_output = self.out(one_output)
            decoder_output_set = torch.cat((decoder_output_set, decoder_output), dim=1)
            decoder_input = decoder_output
        return decoder_output_set

model_GRU = RNN(input_size=3,hidden_size=20,output_size=3)

model_GRU = torch.load("model_GRU")
model_GRU.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_GRU.parameters(),lr=0.001)

def train(epochs):
    train_loss = []
    # minimum = 1e6
    for epoch in range(epochs):
            model_GRU.train()
            output = model_GRU(train_X)
            loss = criterion(output,train_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            model_GRU.eval()
            output_val = model_GRU(val_X)
            val_loss = criterion(output_val, val_Y)

            if epoch % 20 == 0:
                train_loss.append(loss.tolist())
                print(f"epoch:{epoch},train_loss:{loss},val_loss{val_loss}")
    torch.save(model_GRU,"model_GRU")

    return train_loss

# trainLoss = train(300)

datalength = 1
model_GRU.eval().cpu()
test_X = np.loadtxt("1.1_160_0.1_noise.txt")[0:datalength,:]
test_X = np.divide((test_X-a),b).reshape(1,-1,3)
test_X = torch.Tensor(test_X)
result = model_GRU(test_X,predict_length = 300-datalength)
result = result.detach().numpy().reshape(-1,3)
predict_output = np.multiply(result,b) + a
real_output = np.loadtxt("1.1_160_0.1_.txt")[datalength:301,:]

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

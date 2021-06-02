import numpy as np
import torch
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader,TensorDataset
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sklearn import preprocessing


input_a = np.loadtxt("output_1_150_0_t.txt")[0,:]
output_a = np.loadtxt("output_1_150_0_t.txt")[0:301,:]
input_b = np.loadtxt("output_0.9_140_0_t.txt")[0,:]
output_b = np.loadtxt("output_0.9_140_0_t.txt")[0:301,:]

train_X = np.zeros((2,1,3))
train_X[0] = input_a
train_X[1] = input_b
train_Y = np.zeros((2,301,3))
train_Y[0] = output_a
train_Y[1] = output_b
val_X = np.loadtxt("output_1.1_160_0_t.txt")[0 ,:].reshape(1,1,3)
val_Y = np.loadtxt("output_1.1_160_0_t.txt")[1:301,:].reshape(1,300,3)

scale = np.loadtxt("normV_mean_std")
mean = scale[0]
variance = scale[1]

train_X = np.divide((train_X - mean), variance)
train_Y = np.divide((train_Y - mean), variance)
train_X = torch.Tensor(train_X)
train_Y = torch.Tensor(train_Y)
train_X,train_Y =train_X,train_Y

val_X = np.divide((val_X - mean), variance)
# val_X[0,100:,:] = 0
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
        decoder_output_set = torch.zeros([0])
        decoder_output_set = torch.cat((decoder_output_set, decoder_input), dim=1)

        if predict_length == 0:
            return self.out(encoder_outputs)

        for _ in range (predict_length -1):
            one_output, hidden = self.gru (decoder_input,hidden)
            decoder_output = self.out(one_output)
            decoder_output_set = torch.cat((decoder_output_set, decoder_output), dim=1)
            decoder_input = decoder_output
        return decoder_output_set

model_GRU = RNN(input_size=3,hidden_size=16,output_size=3)
model_GRU = torch.load("model_GRU")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_GRU.parameters(),lr=0.001)

def train(epochs):
    train_loss = []
    # minimum = 1e6
    for epoch in range(epochs):
            optimizer.zero_grad()
            model_GRU.train()
            output = model_GRU(train_Y[:,:-1,:],predict_length = 0)
            loss = criterion(output,train_Y[:,1:,:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 20 == 0:
                train_loss.append(loss.tolist())
                print(f"epoch:{epoch},train_loss:{loss}")
    torch.save(model_GRU,"model_GRU")

    return train_loss

# trainLoss = train(1000)


model_GRU.eval()
result = model_GRU(val_X)
result = result.detach().numpy().reshape(300,3)
predict_output = np.multiply(result, variance) + mean
real_output = val_Y.numpy().reshape(300,3)

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

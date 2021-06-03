import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

val_X = np.loadtxt("0.9_150_0_.txt")[:1,:].reshape(1,1,3)
val_Y = np.loadtxt("0.9_150_0_.txt")[1:301,:]

c = np.loadtxt("normV_mean_std")
a = c[0]
b = c[1]

val_X = np.divide((val_X-a),b)
val_X = torch.Tensor(val_X)

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
        for _ in range (predict_length -1):
            one_output, hidden = self.gru (decoder_input,hidden)
            decoder_output = self.out(one_output)
            decoder_output_set = torch.cat((decoder_output_set, decoder_output), dim=1)
            decoder_input = decoder_output
        return decoder_output_set
model_GRU = RNN(input_size=3,hidden_size=20,output_size=3)
model_GRU = torch.load("model_GRU")

model_GRU.eval()
result = model_GRU(val_X)
result = result.detach().numpy().reshape(300,3)
predict_output = np.multiply(result,b) + a
real_output = val_Y

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
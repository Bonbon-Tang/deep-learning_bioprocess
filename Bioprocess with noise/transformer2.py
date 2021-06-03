import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import math
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

step = 1

input_a = np.loadtxt("1.0_150_0_noise.txt")[0,:]
output_a = np.loadtxt("1.0_150_0_noise.txt")[0:301,:]
input_b = np.loadtxt("0.9_140_0_noise.txt")[0,:]
output_b = np.loadtxt("0.9_140_0_noise.txt")[0:301,:]
input_c = np.loadtxt("1.1_160_0.1_noise.txt")[0,:]
output_c = np.loadtxt("1.1_160_0.1_noise.txt")[0:301,:]


train_X = np.zeros((3,1,3))
train_X[0] = input_a
train_X[1] = input_b
train_X[2] = input_c
train_Y = np.zeros((3,301,3))
train_Y[0] = output_a
train_Y[1] = output_b
train_Y[2] = output_c
train_X = train_X.transpose(1,0,2)
train_Y = train_Y.transpose(1,0,2)

val_X = np.loadtxt("1.1_160_0_.txt")[0,:].reshape(1,1,3)
val_Y = np.loadtxt("1.1_160_0_.txt")[0:301,:].reshape(1,301,3)
val_X = val_X.transpose(1,0,2)
val_Y = val_Y.transpose(1,0,2)

c = np.loadtxt("normV_mean_std")
a = c[0]
b = c[1]

train_X = np.divide((train_X-a),b)
train_Y = np.divide((train_Y-a),b)
train_X = torch.Tensor(train_X)
train_Y = torch.Tensor(train_Y)
train_X,train_Y =train_X,train_Y


val_X = np.divide((val_X-a),b)
val_Y = np.divide((val_Y-a),b)
val_X = torch.Tensor(val_X)
val_Y = torch.Tensor(val_Y)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, intoken, outtoken, hidden, nlayers=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        nhead = 4

        self.encoder = nn.Linear(intoken, hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.decoder = nn.Linear(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)

        self.inscale = math.sqrt(intoken)
        self.outscale = math.sqrt(outtoken)

        self.transformer = nn.Transformer(d_model=hidden, nhead=nhead, num_encoder_layers=nlayers,
                                          num_decoder_layers=nlayers, dim_feedforward=hidden, dropout=dropout)
        self.fc_out = nn.Linear(hidden, outtoken)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src, trg):
        #input: time_length,batch,features
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg))

        src_pad_mask = self.make_len_mask(src)
        trg_pad_mask = self.make_len_mask(trg)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)
        output = self.transformer(src, trg, tgt_mask=self.trg_mask)
        # output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask,
        #                           memory_mask=self.memory_mask,
        #                           src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask,
        #                           memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)

        return output

model = TransformerModel(intoken=3, outtoken=3, hidden=32, nlayers=2, dropout=0.1)
model = torch.load("model_transformer")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

def train (epochs):
    for epoch in range (epochs):
        model.train()
        output = model(train_X,train_Y[0:-1])
        loss = criterion(output,train_Y[1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.eval()
        val_output = model(val_X,val_Y[0:-1])
        val_loss = criterion(val_output, val_Y[1:])
        if epoch % 20 == 0:
            print(f"epoch:{epoch},train_loss:{loss},val_loss{val_loss}")

    torch.save(model, "model_transformer")
    return

# _ = train(100)

def evaluate (length,X):
    decoder_input = X[-1].reshape(1,1,3)
    for i in range (length):
        output = model(X,decoder_input)
        output = output[-1].reshape(1,1,3)
        decoder_input = torch.cat((decoder_input,output),dim = 0)
    return decoder_input


test_X = np.loadtxt("0.9_150_0_noise.txt")[0,:]
test_Y = np.loadtxt("0.9_150_0_noise.txt")[0:,:]
test_X = np.divide((test_X-a),b).reshape(1,1,3)
test_Y = np.divide((test_Y-a),b).reshape(1,301,3)
test_X = test_X.transpose(1,0,2)
test_Y = test_Y.transpose(1,0,2)
test_X,test_Y = torch.Tensor(test_X),torch.Tensor(test_Y)


real_output = np.loadtxt("0.9_150_0_.txt")[1:,:]
result = evaluate(300,test_X)
result = result.detach().numpy().reshape(-1,3)
predict_output = np.multiply(result,b) + a

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


import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

x = np.arange(-10, 10, .1)
x_t = torch.tensor(x, requires_grad=True, dtype=torch.float32)
f = x_t**2 + 10*torch.randn(200)

plt.scatter(x_t.detach().numpy(),f.detach().numpy())
plt.show()


# Generate Net class from nn.Module
class Net(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_hidden2, n_output):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(n_inputs, n_hidden)
        self.layer_2 = nn.Linear(n_hidden, n_hidden2)
        self.output = nn.Linear(n_hidden2, n_output)
        
    def forward(self,x):
        out = torch.relu(self.layer_1(x))
        out = torch.relu(self.layer_2(out))
        out = self.output(out)
        return out

model = Net(1, 100, 50, 1)

x_train = x_t.unsqueeze(1)
f_train = f.unsqueeze(1)

loss = nn.MSELoss()
learning_rate = 1e-2
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1000):
    output = model(x_train)
    loss_train = loss(output, f_train)
    optimizer.zero_grad()
    loss_train.backward(retain_graph=True)
    optimizer.step()
    
    if epoch == 1 or epoch % 100 == 0:
        print("epoch: {} and training loss = {}".format(epoch, loss_train.item()))
        plt.scatter(x_t.detach().numpy(),output.data.numpy())
        plt.show()



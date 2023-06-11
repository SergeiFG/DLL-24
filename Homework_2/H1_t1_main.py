import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import math

# %% load and preprocess data
Data_train = torchvision.datasets.MNIST('.', download=True, train=True,  
                                        transform= torchvision.transforms.ToTensor())
Data_test = torchvision.datasets.MNIST('.', download=True, train=False,  
                                        transform= torchvision.transforms.ToTensor())

train = torch.utils.data.DataLoader(Data_train, batch_size=200, shuffle=True)
test = torch.utils.data.DataLoader(Data_test, batch_size=200, shuffle=True)

# %% define model

model = torch.nn.Sequential(torch.nn.Flatten(),
                            torch.nn.Linear(784, 200),
                            torch.nn.BatchNorm1d(200),
                            torch.nn.Sigmoid(),
                            torch.nn.Linear(200, 100),
                            torch.nn.BatchNorm1d(100),
                            torch.nn.ReLU(),
                            torch.nn.Linear(100, 50),
                            torch.nn.BatchNorm1d(50),
                            torch.nn.ReLU(),
                            torch.nn.Linear(50, 10),
                            )

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.CrossEntropyLoss()

# %% training
epoch_num = 10
hist_train = np.array([])
hist_test = np.array([])
acc_train = np.array([])
acc_test = np.array([])

for epoch in range(epoch_num):
    
    ep_hist_train, ep_hist_test = 0, 0
    ep_acc_train, ep_acc_test = 0, 0
    
    model.train()
    for X, Y in train:
        optimizer.zero_grad()
        y_pred = model(X)
        l = loss(y_pred, Y)
        l.backward()
        optimizer.step()
        
        ep_hist_train += l.item()
        ep_acc_train += (y_pred.argmax(dim=1) == Y).sum().item()
    
    model.eval()
    for X, Y in test:        
        y_pred = model(X)
        l = loss(y_pred, Y)
        
        ep_hist_test += l.item()
        ep_acc_test += (y_pred.argmax(dim=1) == Y).sum().item()
    
    hist_train = np.append(hist_train, ep_hist_train/len(Data_train))
    hist_test = np.append(hist_test, ep_hist_test/len(Data_test))
    acc_train = np.append(acc_train, ep_acc_train/len(Data_train)*100)
    acc_test = np.append(acc_test, ep_acc_test/len(Data_test)*100)
    
    if (epoch % 2) == 0:
        print('Done ', epoch, ' of ', epoch_num, ' which is ', 
              math.ceil(epoch/epoch_num*100), '%')
    
plt.subplot(1, 2, 1)
plt.plot(hist_test, label='Loss test')
plt.plot(hist_train, label='Loss train')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_test, label='Accuracy test, %')
plt.plot(acc_train, label='Accuracy train, %')
plt.grid()
plt.legend()
    
    
    
    
    

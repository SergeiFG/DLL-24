import pandas
import numpy as np
import math
from sklearn.utils import shuffle
import torch
import random
import matplotlib.pyplot as plt

# %% importing data

Data_raw = pandas.read_csv('housing.csv')
# Избавимся от строк, содержащих NaN 
nan_rows = Data_raw[Data_raw.isnull().T.any()]
Data_raw.drop(index=nan_rows.index, inplace=True)
# Перемешаем данные на случай, если при разделении на train и test будет перекос по одному из параметров
Data_raw = shuffle(Data_raw)

# %% extracting data from raw dataframe
# Перепишем исходные данные в numpy array, исключив колонки ocean_proximity и median_house_value
col_names = Data_raw.columns.tolist()
col_names.remove('ocean_proximity')
col_names.remove('median_house_value')

X_raw = np.array(Data_raw[col_names].values)
Y_raw = np.array(Data_raw[['median_house_value']].values)

# %% normalize X and Y columns

def norm(arr):
    a = max(arr)
    b = min(arr)
    return ((arr-b)/(a-b)-0.5)*2, a, b


def unnorm(arr, maxes, mins):
    return (arr*0.5 + 0.5)*(maxes - mins) + mins


X = np.zeros(X_raw.shape, dtype=float)
Y = np.zeros(Y_raw.shape, dtype=float)
mins = np.array([])
maxes = np.array([])

for column in range(X_raw.shape[1]):
    X[:, column], a, b  = norm(X_raw[:, column])
    maxes = np.append(maxes, a)
    mins = np.append(mins, b)

Y[:, 0], a, b  = norm(Y_raw[:, 0])
maxes = np.append(maxes, a)
mins = np.append(mins, b)

# %% transforming ocean proximity
# Заменим значения ocean proximity на табличку, где 1 соответсвует каждому знаению
Ocean_prox_vars = list(set(Data_raw.ocean_proximity.values))
Ocean_prox_dict = dict(zip(Ocean_prox_vars, [i for i in range(len(Ocean_prox_vars))]))
Ocean_prox_arr = np.array([[0]*len(Ocean_prox_vars)]*len(Data_raw.ocean_proximity), dtype=float)


for line in range(len(Data_raw)):
    col_num = Ocean_prox_dict[Data_raw.ocean_proximity.values[line]]
    Ocean_prox_arr[line, col_num] = 1  

# Окончательный массив X
X = np.append(X, Ocean_prox_arr, axis=1)


# %% creating train and test datasets

split = 0.8

X_train = torch.tensor(X[:math.floor(0.8*len(X))], dtype=torch.float)
X_test =  torch.tensor(X[math.ceil(0.8*len(X)):], dtype=torch.float)
Y_train =  torch.tensor(Y[:math.floor(0.8*len(X))], dtype=torch.float) 
    
Y_test =  torch.tensor(Y[math.ceil(0.8*len(X)):], dtype=torch.float)  
                      
                    
# %% define neural network

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = indices[i: min(i + batch_size, num_examples)]
        yield features[j, :], labels[j]



model = torch.nn.Sequential(torch.nn.Linear(X_train.shape[1], Y_train.shape[1]))
loss = torch.nn.MSELoss()
trainer = torch.optim.SGD(model.parameters(), lr=0.001)


# %% train model
batch_size = 32
# Инициализуем переменные для
history = np.array([])
accuracy = np.array([])
history = np.append(history, loss(model(X_train), Y_train).detach().numpy())        
accuracy = np.append(accuracy, loss(model(X_test), Y_test).detach().numpy())

num_epochs = 200
for epoch in range(1, num_epochs + 1):
    for X_batch, Y_batch in data_iter(batch_size, X_train, Y_train):
        trainer.zero_grad()
        l = loss(model(X_batch), Y_batch)
        l.backward()
        trainer.step()
        
    history = np.append(history, loss(model(X_train), Y_train).detach().numpy())        
    accuracy = np.append(accuracy, loss(model(X_test), Y_test).detach().numpy())
    
    if epoch % 5 == 0:
        print('Training epoch ', epoch, '/', num_epochs)

# %% visualize loss

plt.plot(history, label='Loss value')
plt.plot(accuracy, label='Accuracy')
plt.legend()
plt.show()

# %% result data
result = model(torch.tensor(X, dtype=torch.float)).detach().numpy()
result = unnorm(result, maxes[-1], mins[-1])
result = np.append(result, np.array(Data_raw.median_house_value.values).reshape((-1,1)), axis=1)    
result = pandas.DataFrame(result, index=Data_raw.index, columns=['Prediction', 'True Value'])

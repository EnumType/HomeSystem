import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import sys
from torch.autograd import Variable
from numpy import array

device = ''
predict_data = []
doTraining = True
doPrediction = True
if __name__ == "__main__":
    if len(sys.argv) >= 4:
        device = str(sys.argv[1])
        doTraining = sys.argv[2].lower() == 'true'
        doPrediction = sys.argv[3].lower() == 'true'

        if doPrediction and len(sys.argv) >= 5:
            args = sys.argv[4].strip('[]').split(',')
            if len(args) == 2:
                weather = int(args[0])
                time = int(args[1])

                predict_array = array([(weather + time)], dtype='int64')
                predict_data.append(predict_array)
            elif doPrediction and len(sys.argv) < 5:
                print('Lenght of PredictionData must be 2. Yours is ', len(args))
                quit()
        elif doPrediction:
            print('USAGE: <DeviceName> <DoTraining> <DoPrediction> <PredictionData>')
            quit()
    else:
        print('USAGE: <DeviceName> <DoTraining> <DoPrediction> [<PredictionData>]')
        quit()

x_list = []
z_list = []
target_list = []
x_data = []
y_data = []
z_data = []
train_data_path = 'AI/data/' + device + '.csv'
modelPath = 'AI/models/' + device + '.pt'

if doTraining:
    for weather, time in zip(
        pd.read_csv(train_data_path, sep=',', chunksize=1, usecols=['Weather']),
        pd.read_csv(train_data_path, sep=',', chunksize=1, usecols=['Time'])
    ):
        weather = weather.values
        time = time.values

        x_list.append(time)
        z_list.append(weather)
    for target in pd.read_csv(train_data_path, sep=',', chunksize=1, usecols=['State']):
        target = target.values
        target_list.append(target)

    for x, z, target in zip(x_list, z_list, target_list):
        x_data.append(x)
        y_data.append(target)
        z_data.append(z)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 100)
        self.leak1 = torch.nn.LeakyReLU()
        self.hidden2 = torch.nn.Linear(100, 200)
        self.leak2 = torch.nn.LeakyReLU()
        self.hidden3 = torch.nn.Linear(200, 100)
        self.leak3 = torch.nn.LeakyReLU()
        self.prediction = torch.nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.leak1(x)
        x = self.hidden2(x)
        x = self.leak2(x)
        x = self.hidden3(x)
        x = self.leak3(x)
        x = self.prediction(x)
        return x


model = Model()
if os.path.isfile(modelPath):
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['state_dict'])

optimizer = optim.Adam(model.parameters(), lr=0.01)

x, y, z = Variable(torch.Tensor(x_data)), Variable(torch.Tensor(y_data)), Variable(torch.Tensor(z_data))

def train(epoch):
    model.train()
    data = x + z
    target = y

    data = (data - data.mean()) / data.std()

    prediction = model(data)
    criterion = torch.nn.MSELoss()
    loss = criterion(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.save({'state_dict': model.state_dict()}, modelPath)
    print('EPOCH: ', epoch, ' LOSS:', loss.data.numpy())


if doTraining:
    for epoch in range(1000):
        train(epoch + 1)

if doPrediction:
    model.eval()
    data = Variable(torch.Tensor(predict_data))
    out = model(data)

    result = int(torch.round(out.data).numpy())

    if result < 0:
        result = 0

    print(result)
    sys.exit(result)

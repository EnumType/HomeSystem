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
            if len(args) == 3:
                weather = int(args[0])
                date = int(args[1])
                time = int(args[2])

                predict_array = array([(date + time + weather)], dtype='int64')
                predict_data.append(predict_array)
            elif doPrediction and len(sys.argv) < 5:
                print('Lenght of PredictionData must be 3. Yours is ', len(args))
                quit()
        elif doPrediction:
            print('USAGE: <DeviceName> <DoTraining> <DoPrediction> <PredictionData>')
            quit()
    else:
        print('USAGE: <DeviceName> <DoTraining> <DoPrediction> [<PredictionData>]')
        quit()

train_list = []
target_list = []
x_data = []
y_data = []
train_data_path = 'AI/data/' + device + '.csv'
modelPath = 'AI/models/' + device + '.pt'

if doTraining:
    for weather, date, time in zip(
        pd.read_csv(train_data_path, sep=',', chunksize=1, usecols=['Weather']),
        pd.read_csv(train_data_path, sep=',', chunksize=1, usecols=['Date']),
        pd.read_csv(train_data_path, sep=',', chunksize=1, usecols=['Time'])
    ):
        weather = weather.values
        date = date.values
        time = time.values

        train_list.append((date + time + weather))
    for target in pd.read_csv(train_data_path, sep=',', chunksize=1, usecols=['State']):
        target = target.values
        target_list.append(target)

    for data, target in zip(train_list, target_list):
        x_data.append(data)
        y_data.append(target)


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


def train(epoch):
    model.train()
    data = Variable(torch.Tensor(x_data))
    target = Variable(torch.Tensor(y_data))

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
    for epoch in range(1, 5000):
        train(epoch)

if doPrediction:
    model.eval()
    data = Variable(torch.Tensor(predict_data))
    out = model(data)

    result = int(torch.round(out.data).numpy())

    if result < 0:
        result = 0

    print(result)
    sys.exit(result)

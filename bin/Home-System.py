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
predict_x = []
predict_z = []
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

                predict_time = array([(time)], dtype='int64')
                predict_weather = array([(weather)], dtype='int64')

                predict_x.append(predict_time)
                predict_z.append(predict_weather)
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

x, y, z = Variable(torch.Tensor(x_data)), Variable(torch.Tensor(y_data)), Variable(torch.Tensor(z_data))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 200)
        self.relu1 = torch.nn.ReLU()
        self.hidden2 = torch.nn.Linear(200, 400)
        self.relu2 = torch.nn.ReLU()
        self.hidden3 = torch.nn.Linear(400, 150)
        self.relu3 = torch.nn.ReLU()
        self.prediction = torch.nn.Linear(150, 1)

    def forward(self, x):
        res = self.hidden1(x)
        res = self.relu1(res)
        res = self.hidden2(res)
        res = self.relu2(res)
        res = self.hidden3(res)
        res = self.relu3(res)
        res = torch.sigmoid(self.prediction(res))
        return res


model = Model()
if os.path.isfile(modelPath):
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['state_dict'])

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCELoss(reduction='mean')


def train(epoch):
    model.train()
    data = x + z
    target = y

    data = (data - data.mean()) / (data.std(unbiased=False) + 1)

    prediction = model(data)
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
    traindata = x + z
    data = Variable(torch.Tensor(predict_x) + torch.Tensor(predict_z))
    data = (data - traindata.mean()) / (traindata.std(unbiased=False) + 1)
    out = model(data)
    print(out.view(-1).data.numpy())
    result = int(torch.round(out.view(-1).data).numpy())

    if result < 0:
        result = 0

    print(result)
    sys.exit(result)

quit()

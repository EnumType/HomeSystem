import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import sys
from torch.autograd import Variable

device = ''
predict_data = []
doTraining = True
doPrediction = True
batch = 64
if __name__ == "__main__":
    if len(sys.argv) >= 4:
        device = str(sys.argv[1])
        doTraining = sys.argv[2].lower() == 'true'
        doPrediction = sys.argv[3].lower() == 'true'

        if doPrediction and len(sys.argv) >= 5:
            args = sys.argv[4].strip('[]').split(',')
            if len(args) == 3:
                predict_data = [int(args[0]), int(args[1]), int(args[2])]
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
train_data = []
target_list = []
train_data_path = 'AI/data/' + device + '.csv'
modelPath = 'AI/models/' + device + '.pt'

if doTraining:
    training_fields = ['Weather', 'Date', 'Time']
    target_fields = ['State']
    temp_data = []
    temp_target = []

    for data in pd.read_csv(train_data_path, sep=',', chunksize=1, usecols=training_fields):
        train_list.append(data.values)

    for target in pd.read_csv(train_data_path, sep=',', chunksize=1, usecols=target_fields):
        target_list.append(target.values + [0, 0])

    for data, target in zip(train_list, target_list):
        temp_data.append(data)
        temp_target.append(target)
        if len(temp_data) >= batch:
            train_data.append((temp_data, temp_target))
            temp_data = []
            temp_target = []
            print('Loaded batch ', len(train_data), 'of ', int(len(train_list) / batch))
            print('Percentage Done: ', len(train_data) / int(len(train_list) / batch) * 100, '%')


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(3, 100)
        self.lin2 = nn.Linear(100, 200)
        self.lin_dropout = nn.Dropout()
        self.lin3 = nn.Linear(200, 2)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = self.lin_dropout(x)
        x = self.lin3(x)
        return x


model = Model()
if os.path.isfile(modelPath):
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['state_dict'])

optimizer = optim.Adam(model.parameters(), lr=0.1)


def train(epoch):
    model.train()
    batch_id = 0
    for data, target in train_data:
        data = Variable(torch.Tensor(data))
        target = Variable(torch.Tensor(target))
        optimizer.zero_grad()
        out = model(data)
        criterion = nn.MSELoss()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        batch_id = batch_id + 1
        print('Train Epoch: {} [{}/{}] {:.0f}%\tLoss: {:.6f}'.format(
            epoch, batch_id * len(data), len(train_data) * batch, 100. * batch_id / len(train_data), loss.data))
        torch.save({'state_dict': model.state_dict()}, modelPath)


if doTraining:
    for epoch in range(1, 500):
        train(epoch)

if doPrediction:
    model.eval()
    tensor = torch.Tensor(predict_data)
    tensor.unsqueeze_(0)
    out = model(tensor)
    print(out.data.max(1, keepdim=True)[1])
    sys.exit(int(out.data.max(1, keepdim=True)[1]))

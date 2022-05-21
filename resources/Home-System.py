import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import sys

device = ''
train_data_path = ''
model_path = ''
do_training = True
do_prediction = True
save_plots = True
time_in_week = 604800
subtract_value = 0
max_epochs = 10000
data_min = 0
data_max = 1
train_data_list = []
time_list = []
target_data_list = []

time_input = 0
light_input = 0
temperature_input = 0
special_input = 0


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = torch.nn.Linear(4, 64)
        self.relu1 = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(64, 512)
        self.relu2 = torch.nn.ReLU()
        self.dense3 = torch.nn.Linear(512, 256)
        self.relu3 = torch.nn.ReLU()
        self.dense4 = torch.nn.Linear(256, 64)
        self.relu4 = torch.nn.ReLU()
        self.dense5 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        x = self.relu3(x)
        x = self.dense4(x)
        x = self.relu4(x)
        x = self.dense5(x)
        return x


model = Model()
optimizer = optim.Adam(model.parameters(), lr=0.0005, amsgrad=True)
criterion = torch.nn.MSELoss(reduction='mean')
losses = []


def load_data():
    global subtract_value
    global data_min
    global data_max

    for time, light, temperature, special in zip(
                pd.read_csv(train_data_path, sep=',', chunksize=1, usecols=['Time']),
                pd.read_csv(train_data_path, sep=',', chunksize=1, usecols=['Light']),
                pd.read_csv(train_data_path, sep=',', chunksize=1, usecols=['Temperature']),
                pd.read_csv(train_data_path, sep=',', chunksize=1, usecols=['Special'])):
        time = time.values[0][0] - subtract_value
        light = light.values[0][0]
        temperature = temperature.values[0][0]
        special = special.values[0][0]

        if subtract_value < 1:
            subtract_value = time
            time = time - subtract_value

        train_data_list.append(torch.tensor([normalize_timestamp(time, time_in_week), light, temperature, special]))
        time_list.append(time)
    for target_value in pd.read_csv(train_data_path, sep=',', chunksize=1, usecols=['State']):
        target_value = target_value.values[0][0]
        target_data_list.append(torch.tensor([target_value]))
    tensor = torch.stack(train_data_list).float()
    data_min = tensor.min()
    data_max = tensor.max()


def load_model_data(filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def normalize_timestamp(input_millis, time_per_period):
    if (input_millis - time_per_period) >= 0:
        result = normalize_timestamp(input_millis - time_per_period, time_per_period)
    else:
        result = input_millis

    return result


def convert_to_array(tensor_list):
    value_list = []
    for t in tensor_list:
        value_list.append(t.detach().item())
    return value_list


def train(epoch):
    model.train()
    data = torch.stack(train_data_list).float()
    data = (data - data_min)/(data_max - data_min) * 2 - 1
    target = torch.stack(target_data_list).float()

    prediction = model(data)
    loss = criterion(prediction, target)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)
    print('EPOCH: ', epoch, ' LOSS:', loss.detach().numpy())

    losses.append(loss.detach().numpy())

    if epoch == max_epochs and save_plots:
        if not os.path.exists("plots/"):
            os.makedirs("plots/")
        create_loss_plot(losses, device + "_loss.png")
        create_model_plot(time_list, prediction.detach().numpy(), device + "_curve.png")


def start_training():
    if os.path.isfile(model_path + '.back'):
        os.remove(model_path + '.back')
    if os.path.isfile(model_path):
        os.rename(model_path, model_path + '.back')

    for i in range(1, max_epochs+1):
        train(i)

    if os.path.isfile(model_path + '.back'):
        os.remove(model_path + '.back')


def predict(time_value, light_value, temperature_value, special_value):
    model.eval()
    input_data = torch.stack([torch.tensor([normalize_timestamp(time_value - subtract_value, time_in_week),
                                            light_value, temperature_value, special_value])]).float()
    input_data = (input_data - data_min)/(data_max - data_min) * 2 - 1
    result = model(input_data).detach().item()
    return result


def create_gif(x_list, target_list, y_list, filename, **kwargs):
    fig = plt.figure()
    ims = []
    plt.gca().set_ylim([-0.5, 1.5])
    plt.gcf().set_size_inches(19.2, 10.8)
    for y in y_list:
        line = plt.plot(x_list, y, ".r")
        ims.append(line)

    plt.plot(x_list, target_list, ".b")
    ani = animation.ArtistAnimation(fig, ims, interval=kwargs.get('interval', 100))
    if kwargs.get('save_plot', save_plots):
        ani.save('plots/' + filename, dpi=1000)
    if kwargs.get('show_plot', False):
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()
    else:
        plt.clf()


def create_model_plot(x_list, result, filename, **kwargs):
    plt.plot(x_list, convert_to_array(target_data_list), ".b")
    plt.plot(x_list, result, ".r")
    if kwargs.get('save_plot', save_plots):
        plt.gcf().set_size_inches(19.2, 10.8)
        plt.savefig('plots/' + filename, dpi=1000)
    if kwargs.get('show_plot', False):
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()
    else:
        plt.clf()


def create_loss_plot(loss_list, filename, **kwargs):
    plt.plot(np.arange(1, max_epochs+1, 1), loss_list, "-b")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if kwargs.get('save_plot', save_plots):
        plt.gcf().set_size_inches(19.2, 10.8)
        plt.savefig('plots/' + filename, dpi=1000)
    if kwargs.get('show_plot', False):
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()
    else:
        plt.clf()


if os.path.isfile(model_path):
    load_model_data(model_path)
elif os.path.isfile(model_path + '.back'):
    load_model_data(model_path + '.back')

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        device = str(sys.argv[1])
        if not device:
            quit(-4)
        train_data_path = 'AI/data/' + device + '.csv'
        model_path = 'AI/models/' + device + '.pt'
        do_training = sys.argv[2].lower() == 'true'
        do_prediction = sys.argv[3].lower() == 'true'

        if do_prediction and len(sys.argv) >= 5:
            args = sys.argv[4].strip('[]').split(',')
            if len(args) == 4:
                time_input = int(args[0])
                light_input = int(args[1])
                temperature_input = int(args[2])
                special_input = int(args[3])
            elif do_prediction and len(args) < 4:
                print('Length of PredictionData must be 4. Yours is ', len(args))
                quit(-3)
        elif do_prediction:
            print('USAGE: <DeviceName> <DoTraining> <DoPrediction> <PredictionData>')
            quit(-2)
    else:
        print('USAGE: <DeviceName> <DoTraining> <DoPrediction> [<PredictionData>]')
        quit(-1)

    load_data()
    if do_training:
        start_training()
    if do_prediction:
        model_out = predict(time_input, light_input, temperature_input, special_input)
        model_result = model_out

        if model_out < 0:
            model_out = 0
        elif model_out > 1:
            model_out = 1

        final_result = round(model_out * 100)
        print(model_result, model_out, final_result)
        sys.exit(final_result)

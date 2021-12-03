import torch
import argparse
from models.LSTNet import Model
from data_utils import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units')
parser.add_argument('--hidRNN', type=int, default=100, help='number of RNN hidden units')
parser.add_argument('--window', type=int, default=264, help='window size')
parser.add_argument('--CNN_kernel', type=int, default=8, help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=0, help='The window size of the highway component')
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=8, metavar='N', help='batch size')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--skip', type=float, default=0)
parser.add_argument('--hidSkip', type=int, default=50)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')
args = parser.parse_args()

def evaluate(dataset, x, y, model, batch_size):
    model.eval()
    mape_0 = 0
    mape_1 = 0
    n_sample = 0
    predict = None
    test = None

    for x, y in dataset.get_batches(x, y, batch_size, False):
        output = model(x)
        if predict is None:
            predict = output
            test = y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, y))

        mape_0 += torch.sum(torch.div(torch.abs(output[:, 0] - y[:, 0]), torch.abs(y[:, 0]) + 1e-8))
        mape_1 += torch.sum(torch.div(torch.abs(output[:, 1] - y[:, 1]), torch.abs(y[:, 1]) + 1e-8))
        n_sample += output.size(0)
    draw_pic(predict.cpu().detach().numpy(), test.cpu().detach().numpy(), dataset.scale)
    return mape_0 / n_sample, mape_1 / n_sample

def draw_pic(predict, test, scale):
    scale = scale.cpu().numpy()
    plt.figure()
    for i in range(0, 2):
        plt.subplot(2, 1, i + 1)
        plt.plot(predict[0: 264, i] * scale[i], label='predict')
        plt.plot(test[0: 264, i] * scale[i], label='test')

    plt.legend()
    plt.show()

device = torch.device('cuda')
data = Data_utility("dataset.csv", 0.6, 0.2, device, 1, 264, 2)
model = Model(args, data).to(device)
checkpoint = torch.load('checkpoint/best_model_59.pth.tar')
model.load_state_dict(checkpoint)
mape_0, mape_1 = evaluate(data, data.test[0], data.test[1], model, args.batch_size)



import time
import argparse
import torch.nn as nn
from data_utils import *
from models.LSTNet import Model


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units')
parser.add_argument('--hidRNN', type=int, default=100, help='number of RNN hidden units')
parser.add_argument('--window', type=int, default=264, help='window size')
parser.add_argument('--CNN_kernel', type=int, default=8, help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=0, help='The window size of the highway component')
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=8, metavar='N', help='batch size')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--skip', type=float, default=0)
parser.add_argument('--hidSkip', type=int, default=50)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')
args = parser.parse_args()


def evaluate(dataset, x, y, model, criterion, batch_size):
    model.eval()
    mape_0 = 0
    mape_1 = 0
    n_sample = 0
    total_loss = 0

    for x, y in dataset.get_batches(x, y, batch_size, False):
        output = model(x)
        mape_0 += torch.sum(torch.div(torch.abs(output[:, 0] - y[:, 0]), torch.abs(y[:, 0]) + 1e-8))
        mape_1 += torch.sum(torch.div(torch.abs(output[:, 1] - y[:, 1]), torch.abs(y[:, 1]) + 1e-8))
        # scale = data.scale.expand(output.size(0), data.pm)
        # loss = criterion(output * scale, y * scale)
        loss = criterion(output, y)
        total_loss += loss.item()
        n_sample += output.size(0)

    return total_loss / n_sample, mape_0 / n_sample, mape_1 / n_sample


def train(data, x, y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_sample = 0
    for x, y in data.get_batches(x, y, batch_size, True):
        model.zero_grad()
        output = model(x)
        # scale = data.scale.expand(output.size(0), data.pm)
        # loss = criterion(output * scale, y * scale)
        loss = criterion(output, y)
        loss.backward()
        optim.step()
        total_loss += loss.item()
        n_sample += output.size(0)
    return total_loss / n_sample

device = torch.device('cuda')
criterion = nn.MSELoss(reduction='sum').to(device)
data = Data_utility("dataset.csv", 0.6, 0.2, device, args.horizon, args.window, args.normalize)


model = Model(args, data).to(device)
optim = torch.optim.Adam(model.parameters(), lr=args.lr)

print('begin training')
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train_loss = train(data, data.train[0], data.train[1], model, criterion, optim, args.batch_size)
    val_loss, mape_0, mape_1 = evaluate(data, data.valid[0], data.valid[1], model, criterion, args.batch_size)

    print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | val_loss {:5.4f} | valid mape_0 {:5.4f} '
    '| valid mape_1 {:5.4f} | '.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, mape_0, mape_1))


    torch.save(model.state_dict(), 'checkpoint/best_model_{}.pth.tar'.format(epoch))



    # if epoch % 5 == 0:
    #     test_loss, mape_test_0, mape_test_1= evaluate(data, data.test[0], data.test[1], model, criterion, args.batch_size)

    #     print("| Test mape_0 {:5.4f} | Test mape_1 {:5.4f} | ".format(mape_test_0, mape_test_1))


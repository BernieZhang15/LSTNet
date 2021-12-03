import torch
import numpy as np
from pandas import read_csv
from torch.autograd import Variable


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize=2):
        self.cuda = cuda
        self.P = window
        self.h = horizon
        self.pm = 2

        self.raw_data = np.array(read_csv(file_name, header=0)).astype(float)
        self.data = np.zeros(self.raw_data.shape)
        self.n, self.m = self.data.shape
        self.scale = np.ones(self.m - 1)

        self._normalized(normalize)
        self._onehot_coding()
        self._split(int(train * self.n), int((train + valid) * self.n))

        self.scale = torch.from_numpy(self.scale).float().to(self.cuda)


    def _onehot_coding(self):
        index = self.data[:, 2].reshape(1, -1).astype(int)
        array = np.eye(8)[index].squeeze(0)
        array = np.delete(array, 0, axis=1)
        self.data = np.concatenate((self.data[:, 0: 2], array), axis=1)
        self.m += 6


    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.
       
        if normalize == 0:
            self.data = self.raw_data
            
        if normalize == 1:
            self.data = self.raw_data / np.max(self.raw_data)
            
        # normalized by the maximum value of each column.
        if normalize == 2:
            for i in range(self.m - 1):
                self.scale[i] = np.max(np.abs(self.raw_data[:, i]))
                self.data[:, i] = self.raw_data[:, i] / self.scale[i]
            self.data[:, self.m - 1] = self.raw_data[:, self.m - 1]

    def _split(self, train, valid):
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set)
        self.valid = self._batchify(valid_set)
        self.test = self._batchify(test_set)

    def _batchify(self, idx_set):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.pm))
        
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.data[start:end, :])
            Y[i, 0: self.pm] = torch.from_numpy(self.data[idx_set[i], 0: self.pm])

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))

        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt].to(self.cuda)
            Y = targets[excerpt].to(self.cuda)
            yield Variable(X), Variable(Y)
            start_idx += batch_size


if __name__ == "__main__":
    data = Data_utility("dataset.csv", 0.6, 0.2, 'cuda', 1, 264, 2)


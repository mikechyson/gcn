import itertools
import os.path as osp
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

# to save processed data
# create a class with namedtuple
Data = namedtuple('Data',
                  ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])


class CoraData(object):
    """
    Data class.

    Used to read, process and save data.
    """
    # cora data url
    download_url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    filenames = ['ind.cora.{}'.format(name) for name in ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root='cora', rebuild=False):
        # where to store data locally
        self.data_root = data_root
        # where to store processed data
        save_file = osp.join(self.data_root, 'processed_cora.pkl')

        # if the processed data exist, just read it
        if osp.exists(save_file) and not rebuild:
            print('Using Cached file: {}'.format(save_file))
            self._data = pickle.load(open(save_file, 'rb'))
        # download, process and save the data
        else:
            self._data = self.process_data()
            with open(save_file, 'wb') as f:
                pickle.dump(self._data, f)
            print('Cached file: {}'.format(save_file))

    @property
    def data(self):
        return self._data

    def process_data(self):
        print('[INFO] processing data...')
        _, tx, allx, y, ty, ally, graph, test_index = \
            [self.read_data(osp.join(self.data_root, 'raw', name)) for name in self.filenames]
        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        sorted_test_index = sorted(test_index)

        x = np.concatenate((allx, tx), axis=0)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

        adjacency = self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x,
                    y=y,
                    adjacency=adjacency,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)

        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.array(edge_index)
        adjacency = sp.coo_matrix(
            (
                np.ones(len(edge_index)),
                (edge_index[:, 0],
                 edge_index[:, 1])
            ),
            shape=(num_nodes, num_nodes),
            dtype='float32'
        )
        return adjacency

    @staticmethod
    def read_data(path):
        name = osp.basename(path)
        if name == 'ind.cora.test.index':
            out = np.genfromtxt(path, dtype='int64')
            return out
        else:
            out = pickle.load(open(path, 'rb'), encoding='latin1')
            out = out.toarray() if hasattr(out, 'toarray') else out
            return out


class GraphConvolution(nn.Module):
    """
    Graph convolution.
    X = L_sym * X * W
    """

    def __init__(self, input_dim, output_dim, use_bias=True):
        """
        Set up and init the graph convolution network.

        :param input_dim: input dimension
        :param output_dim: output dimension
        :param use_bias: use bias or not
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))

        # set up the bias parameter
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        # init the parameters
        self.init_parameters()

    def init_parameters(self):
        """
        Init parameters in the network.
        :return:
        """
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """
        Graph convolution.
        X = L_sym * X * W

        :param adjacency: adjacency matrix
        :param input_feature:
        :return:
        """
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)

        if self.use_bias:
            output += self.bias

        return output


class GcnNet(nn.Module):
    def __init__(self, input_dim=1433):
        """
        Init two graph convolution layer without activation.

        :param input_dim: input dimension
        """
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 16)  # hidden layer with 16
        self.gcn2 = GraphConvolution(16, 7)

    def forward(self, adjacency, feature):
        """
        A graph network with two graph convolution.

        :param adjacency: adjacency matrix
        :param feature: features of the nodes
        :return:
        """
        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits


def normalization(adjacency):
    """
    Compute L = D^(-0.5) * (A + I) * D^(-0.5).

    :param adjacency:
    :return:
    """
    # construct a unit matrix
    I = sp.eye(adjacency.shape[0])
    adjacency += I

    degree = np.array(adjacency.sum(1))  # degree matrix
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()


def train():
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(epoches):
        logits = model(tensor_adjacency, tensor_x)  # forward
        train_mask_logits = logits[tensor_train_mask]
        loss = criterion(train_mask_logits, train_y)
        opt.zero_grad()
        loss.backward()  # backward to compute the gradient
        opt.step()  # update the gradient
        train_acc = test(tensor_train_mask)
        val_acc = test(tensor_val_mask)

        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print('Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc: {:.4f}'.
              format(epoch, loss.item(), train_acc.item(), val_acc.item()))
    return loss_history, val_acc_history


def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuracy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuracy


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history, c='red')
    plt.ylabel('Loss')

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history, c='blue')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('Validation Accuracy')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


if __name__ == '__main__':
    # hyperparameters
    lr = 0.1  # learning rate
    wd = 5e-4  # weight decay
    epoches = 100

    # model
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = GcnNet().to(device)

    # loss
    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # load data
    dataset = CoraData().data

    # process x and y
    x = dataset.x / dataset.x.sum(1, keepdims=True)  # normalization
    tensor_x = torch.from_numpy(x).to(device)
    tensor_y = torch.from_numpy(dataset.y).to(device)

    # process masks
    tensor_train_mask = torch.from_numpy(dataset.train_mask).to(device)
    tensor_val_mask = torch.from_numpy(dataset.val_mask).to(device)
    tensor_test_mask = torch.from_numpy(dataset.test_mask).to(device)

    # normalize adjacency
    normalize_adjacency = normalization(dataset.adjacency)

    # generate indices of the adjacency matrix
    indices = torch.from_numpy(np.asarray(
        [normalize_adjacency.row,
         normalize_adjacency.col]
    )).long()

    # values of the adjacency matrix
    values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))

    # form the tensor adjacency
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, (2708, 2708)).to(device)

    loss, val_acc = train()
    plot_loss_with_acc(loss, val_acc)

import sys
from xml.parsers.expat import model

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    inner_seq = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    return nn.Sequential(
        nn.Residual(inner_seq),
        nn.ReLU(),
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)

    if opt is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_err = 0.0

    for X, y in dataloader:
        B = X.shape[0]

        X = X.reshape((B, -1))

        if opt is not None:
            opt.reset_grad()

        logits = model(X)
        loss, err = loss_err(logits, y)

        if opt is not None:
            loss.backward()
            opt.step()

        total_loss += loss.numpy().item() * B
        total_err += err * B

    N = len(dataloader.dataset)

    return total_err / N, total_loss / N

def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    dataloader_train = ndl.data.DataLoader(
        ndl.data.MNISTDataset(
            os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
            os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    dataloader_test = ndl.data.DataLoader(
        ndl.data.MNISTDataset(
            os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
            os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    model = MLPResNet(28 * 28, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        train_err, train_loss = epoch(dataloader_train, model, opt)
        test_err, test_loss = epoch(dataloader_test, model)
    return (train_err, train_loss, test_err, test_loss)
    ### END YOUR SOLUTION

def loss_err(h, y):
    """Helper function to compute both loss and error"""
    return nn.SoftmaxLoss()(h, y), np.mean(h.numpy().argmax(axis=1) != y.numpy())

if __name__ == "__main__":
    train_mnist(data_dir="../data")

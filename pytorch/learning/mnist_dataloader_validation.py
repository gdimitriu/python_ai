from pathlib import Path
import requests
import argparse
import pickle
import gzip
from matplotlib import pyplot
from torch import optim
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np

loss_func = F.cross_entropy

FILENAME = "mnist.pkl.gz"
PATH = Path("data")


def build_arg_parser():
    parser = argparse.ArgumentParser(description='MNIST data')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
                        default='../../data', help='Directory for storing data')
    return parser


def download_mnist(path):
    global PATH
    DATA_PATH = Path(path)
    PATH = DATA_PATH
    URL = "https://github.com/pytorch/tutorials/raw/main/_static/"

    if not (DATA_PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (DATA_PATH / FILENAME).open("wb").write(content)


class Mnist_Logistic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def get_model(lr):
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(model, epochs, train_dl, valid_dl, opt):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    download_mnist(args.input_dir)
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
    # ``pyplot.show()`` only if not on Colab
    try:
        import google.colab
    except ImportError:
        pyplot.show()
    print(x_train.shape)

    # convert our data
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    n, c = x_train.shape
    print(x_train, y_train)
    print(x_train.shape)
    print(y_train.min(), y_train.max())

    bs = 64  # batch size

    xb = x_train[0:bs]  # a mini-batch from x
    lr = 0.5  # learning rate
    model, opt = get_model(lr)
    preds = model(xb)  # predictions
    print(preds[0], preds.shape)
    yb = y_train[0:bs]

    print(loss_func(preds, yb))
    print(accuracy(preds, yb))

    epochs = 2  # how many epochs to train for
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    model, opt = get_model(lr)
    fit(model, epochs, train_dl, valid_dl, opt)

    print("loss funct and accuracy")
    print(loss_func(model(xb), yb), accuracy(model(xb), yb))

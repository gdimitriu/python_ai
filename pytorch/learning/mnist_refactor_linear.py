from pathlib import Path
import requests
import argparse
import pickle
import gzip
from matplotlib import pyplot
import math
import torch
import torch.nn.functional as F

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


def fit(model, epochs, lr, bs, x_train, y_train):
    from IPython.core.debugger import set_trace
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            # set_trace()
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()


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
    model = Mnist_Logistic()
    preds = model(xb)  # predictions
    print(preds[0], preds.shape)
    yb = y_train[0:bs]

    print(loss_func(preds, yb))
    print(accuracy(preds, yb))

    lr = 0.5  # learning rate
    epochs = 2  # how many epochs to train for

    fit(model, epochs, lr, bs, x_train, y_train)

    print("loss funct and accuracy")
    print(loss_func(model(xb), yb), accuracy(model(xb), yb))

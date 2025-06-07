import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import argparse

def build_arg_parser():
    parser = argparse.ArgumentParser(description='transforms')
    parser.add_argument('--data-directory', dest='data_directory', type=str,
                        default='.', help='data root')
    return parser

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    ds = datasets.FashionMNIST(
        root=args.data_directory,
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )



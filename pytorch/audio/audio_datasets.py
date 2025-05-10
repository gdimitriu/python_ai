# https://pytorch.org/audio/stable/tutorials/audio_datasets_tutorial.html
import torch
import torchaudio
import os
import matplotlib.pyplot as plt
import argparse

print(torch.__version__)
print(torchaudio.__version__)


def build_arg_parser():
    parser = argparse.ArgumentParser(description='audio datasets')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
                        default='../../data', help='Directory for storing data')
    return parser


def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    figure, ax = plt.subplots()
    ax.specgram(waveform[0], Fs=sample_rate)
    figure.suptitle(title)
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    _SAMPLE_DIR = args.input_dir + "/_assets"
    YESNO_DATASET_PATH = os.path.join(_SAMPLE_DIR, "yes_no")
    os.makedirs(YESNO_DATASET_PATH, exist_ok=True)
    dataset = torchaudio.datasets.YESNO(YESNO_DATASET_PATH, download=True)
    i = 1
    waveform, sample_rate, label = dataset[i]
    plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
    # IPython.display.Audio(waveform, rate=sample_rate)
    i = 3
    waveform, sample_rate, label = dataset[i]
    plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
    # IPython.display.Audio(waveform, rate=sample_rate)
    i = 5
    waveform, sample_rate, label = dataset[i]
    plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
    # IPython.display.Audio(waveform, rate=sample_rate)

# https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html
import torch
import torchaudio
import io
import os
import tarfile
import tempfile

import boto3
import matplotlib.pyplot as plt
import requests
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio
from torchaudio.utils import download_asset

print(torch.__version__)
print(torchaudio.__version__)


def _hide_seek(obj):
    class _wrapper:
        def __init__(self, obj):
            self.obj = obj

        def read(self, n):
            return self.obj.read(n)

    return _wrapper(obj)


def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
    figure.suptitle("waveform")
    plt.show()


def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
    figure.suptitle(title)
    plt.show()


def inspect_file(path):
    print("-" * 10)
    print("Source:", path)
    print("-" * 10)
    print(f" - File size: {os.path.getsize(path)} bytes")
    print(f" - {torchaudio.info(path)}")
    print()


if __name__ == "__main__":
    SAMPLE_GSM = download_asset("tutorial-assets/steam-train-whistle-daniel_simon.gsm")
    SAMPLE_WAV = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
    SAMPLE_WAV_8000 = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav")
    metadata = torchaudio.info(SAMPLE_WAV)
    print(metadata)
    metadata = torchaudio.info(SAMPLE_GSM)
    print(metadata)
    url = "https://download.pytorch.org/torchaudio/tutorial-assets/steam-train-whistle-daniel_simon.wav"
    metadata = torchaudio.info(url)
    print(metadata)
    waveform, sample_rate = torchaudio.load(SAMPLE_WAV)
    plot_waveform(waveform, sample_rate)
    plot_specgram(waveform, sample_rate)
    # Load audio data as HTTP request
    url = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
    waveform, sample_rate = torchaudio.load(url)
    plot_specgram(waveform, sample_rate, title="HTTP datasource")
    # Load audio from tar file
    tar_path = download_asset("tutorial-assets/VOiCES_devkit.tar.gz")
    tar_item = "VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
    with tarfile.open(tar_path, mode="r") as tarfile_:
        fileobj = tarfile_.extractfile(tar_item)
        waveform, sample_rate = torchaudio.load(fileobj)
    plot_specgram(waveform, sample_rate, title="TAR file")
    # Load audio from S3
    # does not work you have to give file
    # bucket = "pytorch-tutorial-assets"
    # key = "VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
    # client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    # response = client.get_object(Bucket=bucket, Key=key)
    # waveform, sample_rate = torchaudio.load(_hide_seek(response["Body"]))
    # plot_specgram(waveform, sample_rate, title="From S3")
    # Illustration of two different decoding methods.
    # The first one will fetch all the data and decode them, while
    # the second one will stop fetching data once it completes decoding.
    # The resulting waveforms are identical.

    frame_offset, num_frames = 16000, 16000  # Fetch and decode the 1 - 2 seconds

    url = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
    print("Fetching all the data...")
    waveform1, sample_rate1 = torchaudio.load(url)
    waveform1 = waveform1[:, frame_offset: frame_offset + num_frames]

    waveform, sample_rate = torchaudio.load(SAMPLE_WAV)
    with tempfile.TemporaryDirectory() as tempdir:
        path = f"{tempdir}/save_example_default.wav"
        torchaudio.save(path, waveform, sample_rate)
        inspect_file(path)
    with tempfile.TemporaryDirectory() as tempdir:
        path = f"{tempdir}/save_example_PCM_S16.wav"
        torchaudio.save(path, waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)
        inspect_file(path)

    formats = [
        "flac",
        # "vorbis",
        # "sph",
        # "amb",
        # "amr-nb",
        # "gsm",
    ]
    waveform, sample_rate = torchaudio.load(SAMPLE_WAV_8000)
    with tempfile.TemporaryDirectory() as tempdir:
        for format in formats:
            path = f"{tempdir}/save_example.{format}"
            torchaudio.save(path, waveform, sample_rate, format=format)
            inspect_file(path)

    waveform, sample_rate = torchaudio.load(SAMPLE_WAV)

    # Saving to bytes buffer
    buffer_ = io.BytesIO()
    torchaudio.save(buffer_, waveform, sample_rate, format="wav")

    buffer_.seek(0)
    print(buffer_.read(16))
    
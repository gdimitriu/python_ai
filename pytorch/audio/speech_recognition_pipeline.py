# https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html
import torch
import torchaudio
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset
import argparse
import pathlib

print(torch.__version__)
print(torchaudio.__version__)

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


def build_arg_parser():
    parser = argparse.ArgumentParser(description='speech recognition pipeline')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
                        default='../../data', help='Directory for storing data')
    return parser


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    SPEECH_FILE = download_asset(key="tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav",
                                 path=pathlib.Path(
                                     args.input_dir + "/audio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"))
    print(SPEECH_FILE)
    # CREATING A PIPELINE
    # First, we will create a Wav2Vec2 model that performs the feature extraction and the classification.
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

    print("Sample Rate:", bundle.sample_rate)
    print("Labels:", bundle.get_labels())

    model = bundle.get_model().to(device)
    print(model.__class__)

    # LOAD THE DATA
    waveform, sample_rate = torchaudio.load(SPEECH_FILE)
    waveform = waveform.to(device)

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    # Extracting acoustic features
    with torch.inference_mode():
        features, _ = model.extract_features(waveform)
    fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
    for i, feats in enumerate(features):
        ax[i].imshow(feats[0].cpu(), interpolation="nearest")
        ax[i].set_title(f"Feature from transformer layer {i + 1}")
        ax[i].set_xlabel("Feature dimension")
        ax[i].set_ylabel("Frame (time-axis)")
    fig.tight_layout()
    plt.show()

    # Feature classification
    with torch.inference_mode():
        emission, _ = model(waveform)
    plt.imshow(emission[0].cpu().T, interpolation="nearest")
    plt.title("Classification result")
    plt.xlabel("Frame (time-axis)")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.show()
    print("Class labels:", bundle.get_labels())

    # Now create the decoder object and decode the transcript.
    decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    transcript = decoder(emission[0])
    print(transcript)

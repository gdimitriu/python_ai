# https://pytorch.org/audio/stable/tutorials/tacotron2_pipeline_tutorial.html
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
import torch
import torchaudio
import matplotlib.pyplot as plt

torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(torch.__version__)
print(torchaudio.__version__)
print(device)
symbols = "_-!'(),.:;? abcdefghijklmnopqrstuvwxyz"


def text_to_sequence(text):
    text = text.lower()
    return [look_up[s] for s in text if s in symbols]


def plot():
    fig, ax = plt.subplots(3, 1)
    for i in range(3):
        with torch.inference_mode():
            spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
        print(spec[0].shape)
        ax[i].imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")
    plt.show()


def plot_waveforms(waveforms, spec, sample_rate):
    waveforms = waveforms.cpu().detach()

    fig, [ax1, ax2] = plt.subplots(2, 1)
    ax1.plot(waveforms[0])
    ax1.set_xlim(0, waveforms.size(-1))
    ax1.grid(True)
    ax2.imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")
    plt.show()
    # return IPython.display.Audio(waveforms[0:1], rate=sample_rate)
    return


if __name__ == "__main__":
    # Character-based encoding
    look_up = {s: i for i, s in enumerate(symbols)}
    symbols = set(symbols)
    text = "Hello world! Text to speech!"
    print(text_to_sequence(text))
    processor = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH.get_text_processor()

    text = "Hello world! Text to speech!"
    processed, lengths = processor(text)

    print(processed)
    print(lengths)

    text = "Hello world! Text to speech!"
    print(text_to_sequence(text))
    print([processor.tokens[i] for i in processed[0, : lengths[0]]])

    # Phoneme-based encoding
    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

    processor = bundle.get_text_processor()

    text = "Hello world! Text to speech!"
    with torch.inference_mode():
        processed, lengths = processor(text)

    print(processed)
    print(lengths)
    print([processor.tokens[i] for i in processed[0, : lengths[0]]])

    # Spectrogram Generation
    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
    processor = bundle.get_text_processor()
    tacotron2 = bundle.get_tacotron2().to(device)

    text = "Hello world! Text to speech!"

    with torch.inference_mode():
        processed, lengths = processor(text)
        processed = processed.to(device)
        lengths = lengths.to(device)
        spec, _, _ = tacotron2.infer(processed, lengths)

    _ = plt.imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")
    plt.show()
    plot()

    # Waveform Generation
    # WaveRNN Vocoder
    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
    processor = bundle.get_text_processor()
    tacotron2 = bundle.get_tacotron2().to(device)
    vocoder = bundle.get_vocoder().to(device)

    text = "Hello world! Text to speech!"

    with torch.inference_mode():
        processed, lengths = processor(text)
        processed = processed.to(device)
        lengths = lengths.to(device)
        spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
        waveforms, lengths = vocoder(spec, spec_lengths)
    plot_waveforms(waveforms, spec, vocoder.sample_rate)

    # Griffin-Lim Vocoder
    bundle = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH

    processor = bundle.get_text_processor()
    tacotron2 = bundle.get_tacotron2().to(device)
    vocoder = bundle.get_vocoder().to(device)

    with torch.inference_mode():
        processed, lengths = processor(text)
        processed = processed.to(device)
        lengths = lengths.to(device)
        spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
    waveforms, lengths = vocoder(spec, spec_lengths)
    plot_waveforms(waveforms, spec, vocoder.sample_rate)

    # Waveglow Vocoder
    # Workaround to load model mapped on GPU
    # https://stackoverflow.com/a/61840832
    waveglow = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub",
        "nvidia_waveglow",
        model_math="fp32",
        pretrained=False,
    )
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://api.ngc.nvidia.com/v2/models/nvidia/waveglowpyt_fp32/versions/1/files/nvidia_waveglowpyt_fp32_20190306.pth",
        # noqa: E501
        progress=False,
        map_location=device,
    )
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}

    waveglow.load_state_dict(state_dict)
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to(device)
    waveglow.eval()

    with torch.no_grad():
        waveforms = waveglow.infer(spec)
    plot_waveforms(waveforms, spec, 22050)

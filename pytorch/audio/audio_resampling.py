# https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html
import io

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import math
import timeit

import librosa
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import resampy

print(torch.__version__)
print(torchaudio.__version__)

DEFAULT_OFFSET = 201


def _get_log_freq(sample_rate, max_sweep_rate, offset):
    """Get freqs evenly spaced out in log-scale, between [0, max_sweep_rate // 2]

    offset is used to avoid negative infinity `log(offset + x)`.

    """
    start, stop = math.log(offset), math.log(offset + max_sweep_rate // 2)
    return torch.exp(torch.linspace(start, stop, sample_rate, dtype=torch.double)) - offset


def _get_inverse_log_freq(freq, sample_rate, offset):
    """Find the time where the given frequency is given by _get_log_freq"""
    half = sample_rate // 2
    return sample_rate * (math.log(1 + freq / offset) / math.log(1 + half / offset))


def _get_freq_ticks(sample_rate, offset, f_max):
    # Given the original sample rate used for generating the sweep,
    # find the x-axis value where the log-scale major frequency values fall in
    times, freq = [], []
    for exp in range(2, 5):
        for v in range(1, 10):
            f = v * 10 ** exp
            if f < sample_rate // 2:
                t = _get_inverse_log_freq(f, sample_rate, offset) / sample_rate
                times.append(t)
                freq.append(f)
    t_max = _get_inverse_log_freq(f_max, sample_rate, offset) / sample_rate
    times.append(t_max)
    freq.append(f_max)
    return times, freq


def get_sine_sweep(sample_rate, offset=DEFAULT_OFFSET):
    max_sweep_rate = sample_rate
    freq = _get_log_freq(sample_rate, max_sweep_rate, offset)
    delta = 2 * math.pi * freq / sample_rate
    cummulative = torch.cumsum(delta, dim=0)
    signal = torch.sin(cummulative).unsqueeze(dim=0)
    return signal


def plot_sweep(
        waveform,
        sample_rate,
        title,
        max_sweep_rate=48000,
        offset=DEFAULT_OFFSET,
):
    x_ticks = [100, 500, 1000, 5000, 10000, 20000, max_sweep_rate // 2]
    y_ticks = [1000, 5000, 10000, 20000, sample_rate // 2]

    time, freq = _get_freq_ticks(max_sweep_rate, offset, sample_rate // 2)
    freq_x = [f if f in x_ticks and f <= max_sweep_rate // 2 else None for f in freq]
    freq_y = [f for f in freq if f in y_ticks and 1000 <= f <= sample_rate // 2]

    figure, axis = plt.subplots(1, 1)
    _, _, _, cax = axis.specgram(waveform[0].numpy(), Fs=sample_rate)
    plt.xticks(time, freq_x)
    plt.yticks(freq_y, freq_y)
    axis.set_xlabel("Original Signal Frequency (Hz, log scale)")
    axis.set_ylabel("Waveform Frequency (Hz)")
    axis.xaxis.grid(True, alpha=0.67)
    axis.yaxis.grid(True, alpha=0.67)
    figure.suptitle(f"{title} (sample rate: {sample_rate} Hz)")
    plt.colorbar(cax)
    plt.show()


def benchmark_resample_functional(
        waveform,
        sample_rate,
        resample_rate,
        lowpass_filter_width=6,
        rolloff=0.99,
        resampling_method="sinc_interp_hann",
        beta=None,
        iters=5,
):
    return (
            timeit.timeit(
                stmt="""
torchaudio.functional.resample(
    waveform,
    sample_rate,
    resample_rate,
    lowpass_filter_width=lowpass_filter_width,
    rolloff=rolloff,
    resampling_method=resampling_method,
    beta=beta,
)
        """,
                setup="import torchaudio",
                number=iters,
                globals=locals(),
            )
            * 1000
            / iters
    )


def benchmark_resample_transforms(
        waveform,
        sample_rate,
        resample_rate,
        lowpass_filter_width=6,
        rolloff=0.99,
        resampling_method="sinc_interp_hann",
        beta=None,
        iters=5,
):
    return (
            timeit.timeit(
                stmt="resampler(waveform)",
                setup="""
import torchaudio

resampler = torchaudio.transforms.Resample(
    sample_rate,
    resample_rate,
    lowpass_filter_width=lowpass_filter_width,
    rolloff=rolloff,
    resampling_method=resampling_method,
    dtype=waveform.dtype,
    beta=beta,
)
resampler.to(waveform.device)
        """,
                number=iters,
                globals=locals(),
            )
            * 1000
            / iters
    )


def benchmark_resample_librosa(
        waveform,
        sample_rate,
        resample_rate,
        res_type=None,
        iters=5,
):
    waveform_np = waveform.squeeze().numpy()
    return (
            timeit.timeit(
                stmt="""
librosa.resample(
    waveform_np,
    orig_sr=sample_rate,
    target_sr=resample_rate,
    res_type=res_type,
)
        """,
                setup="import librosa",
                number=iters,
                globals=locals(),
            )
            * 1000
            / iters
    )


def benchmark(sample_rate, resample_rate):
    times, rows = [], []
    waveform = get_sine_sweep(sample_rate).to(torch.float32)

    args = (waveform, sample_rate, resample_rate)

    # sinc 64 zero-crossings
    f_time = benchmark_resample_functional(*args, lowpass_filter_width=64)
    t_time = benchmark_resample_transforms(*args, lowpass_filter_width=64)
    times.append([None, f_time, t_time])
    rows.append("sinc (width 64)")

    # sinc 6 zero-crossings
    f_time = benchmark_resample_functional(*args, lowpass_filter_width=16)
    t_time = benchmark_resample_transforms(*args, lowpass_filter_width=16)
    times.append([None, f_time, t_time])
    rows.append("sinc (width 16)")

    # kaiser best
    kwargs = {
        "lowpass_filter_width": 64,
        "rolloff": 0.9475937167399596,
        "resampling_method": "sinc_interp_kaiser",
        "beta": 14.769656459379492,
    }
    lib_time = benchmark_resample_librosa(*args, res_type="kaiser_best")
    f_time = benchmark_resample_functional(*args, **kwargs)
    t_time = benchmark_resample_transforms(*args, **kwargs)
    times.append([lib_time, f_time, t_time])
    rows.append("kaiser_best")

    # kaiser fast
    kwargs = {
        "lowpass_filter_width": 16,
        "rolloff": 0.85,
        "resampling_method": "sinc_interp_kaiser",
        "beta": 8.555504641634386,
    }
    lib_time = benchmark_resample_librosa(*args, res_type="kaiser_fast")
    f_time = benchmark_resample_functional(*args, **kwargs)
    t_time = benchmark_resample_transforms(*args, **kwargs)
    times.append([lib_time, f_time, t_time])
    rows.append("kaiser_fast")

    df = pd.DataFrame(times, columns=["librosa", "functional", "transforms"], index=rows)
    return df


def plot(df):
    print(df.round(2))
    ax = df.plot(kind="bar")
    plt.ylabel("Time Elapsed [ms]")
    plt.xticks(rotation=0, fontsize=10)
    for cont, col, color in zip(ax.containers, df.columns, mcolors.TABLEAU_COLORS):
        label = ["N/A" if v != v else str(v) for v in df[col].round(2)]
        ax.bar_label(cont, labels=label, color=color, fontweight="bold", fontsize="x-small")
    plt.show()


if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    sample_rate = 48000
    waveform = get_sine_sweep(sample_rate)

    plot_sweep(waveform, sample_rate, title="Original Waveform")

    # Audio(waveform.numpy()[0], rate=sample_rate)
    resample_rate = 32000
    resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)

    plot_sweep(resampled_waveform, resample_rate, title="Resampled Waveform")
    # Audio(resampled_waveform.numpy()[0], rate=resample_rate)

    # LOWPASS FILTER
    sample_rate = 48000
    resample_rate = 32000

    resampled_waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=6)
    plot_sweep(resampled_waveform, resample_rate, title="lowpass_filter_width=6")

    resampled_waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=128)
    plot_sweep(resampled_waveform, resample_rate, title="lowpass_filter_width=128")

    # ROLLOFF
    sample_rate = 48000
    resample_rate = 32000

    resampled_waveform = F.resample(waveform, sample_rate, resample_rate, rolloff=0.99)
    plot_sweep(resampled_waveform, resample_rate, title="rolloff=0.99")
    resampled_waveform = F.resample(waveform, sample_rate, resample_rate, rolloff=0.8)
    plot_sweep(resampled_waveform, resample_rate, title="rolloff=0.8")

    # WINDOW FUNCTION
    sample_rate = 48000
    resample_rate = 32000

    resampled_waveform = F.resample(waveform, sample_rate, resample_rate, resampling_method="sinc_interp_hann")
    plot_sweep(resampled_waveform, resample_rate, title="Hann Window Default")
    resampled_waveform = F.resample(waveform, sample_rate, resample_rate, resampling_method="sinc_interp_kaiser")
    plot_sweep(resampled_waveform, resample_rate, title="Kaiser Window Default")

    # Comparison against librosa
    sample_rate = 48000
    resample_rate = 32000
    ## Kaiser_best
    resampled_waveform = F.resample(
        waveform,
        sample_rate,
        resample_rate,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )
    plot_sweep(resampled_waveform, resample_rate, title="Kaiser Window Best (torchaudio)")
    librosa_resampled_waveform = torch.from_numpy(
        librosa.resample(waveform.squeeze().numpy(), orig_sr=sample_rate, target_sr=resample_rate,
                         res_type="kaiser_best")
    ).unsqueeze(0)
    plot_sweep(librosa_resampled_waveform, resample_rate, title="Kaiser Window Best (librosa)")
    mse = torch.square(resampled_waveform - librosa_resampled_waveform).mean().item()
    print("torchaudio and librosa kaiser best MSE:", mse)

    # KAISER_FAST
    resampled_waveform = F.resample(
        waveform,
        sample_rate,
        resample_rate,
        lowpass_filter_width=16,
        rolloff=0.85,
        resampling_method="sinc_interp_kaiser",
        beta=8.555504641634386,
    )
    plot_sweep(resampled_waveform, resample_rate, title="Kaiser Window Fast (torchaudio)")
    librosa_resampled_waveform = torch.from_numpy(
        librosa.resample(waveform.squeeze().numpy(), orig_sr=sample_rate, target_sr=resample_rate,
                         res_type="kaiser_fast")
    ).unsqueeze(0)
    plot_sweep(librosa_resampled_waveform, resample_rate, title="Kaiser Window Fast (librosa)")
    mse = torch.square(resampled_waveform - librosa_resampled_waveform).mean().item()
    print("torchaudio and librosa kaiser fast MSE:", mse)

    # PERFORMANCE BENCHAMARKING
    print(f"torchaudio: {torchaudio.__version__}")
    print(f"librosa: {librosa.__version__}")
    print(f"resampy: {resampy.__version__}")
    df = benchmark(48_000, 44_100)
    print("Downsample (48 -> 44.1 kHz)")
    print(df)
    plot(df)
    df = benchmark(16_000, 8_000)
    print("Downsample (16 -> 8 kHz)")
    print(df)
    plot(df)
    df = benchmark(44_100, 48_000)
    print("Upsample (44.1 -> 48 kHz)")
    print(df)
    plot(df)
    df = benchmark(8_000, 16_000)
    print("Upsample (8 -> 16 kHz)")
    print(df)
    plot(df)

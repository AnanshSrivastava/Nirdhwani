import argparse
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft


def wiener_denoise(
    noisy,
    sr,
    noise_seconds=0.5,
    n_fft=512,
    hop_length=128,
    gain_floor=0.05,
):
    """Denoise a mono waveform using a simple Wiener filter."""

    noisy = np.asarray(noisy, dtype=np.float32)

    # Make sure the signal is mono.
    if noisy.ndim > 1:
        noisy = np.mean(noisy, axis=1)

    # STFT
    _, _, Y = stft(
        noisy,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
        padded=True,
    )

    power = np.abs(Y) ** 2

    # Estimate noise from the beginning of the recording.
    # This assumes the first `noise_seconds` contains mostly noise.
    noise_frames = max(1, int(np.ceil(noise_seconds * sr / hop_length)))
    noise_frames = min(noise_frames, power.shape[1])

    noise_power = np.mean(power[:, :noise_frames], axis=1, keepdims=True)

    # Estimate speech power using noisy power minus estimated noise power.
    speech_power = np.maximum(power - noise_power, 0.0)

    # Wiener gain:
    # G = Ps / (Ps + Pn)
    gain = speech_power / (speech_power + noise_power + 1e-10)

    # Avoid complete spectral holes, which can sound harsh.
    gain = np.maximum(gain, gain_floor)

    # Apply gain and reconstruct.
    enhanced_stft = gain * Y

    _, enhanced = istft(
        enhanced_stft,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        input_onesided=True,
        boundary=True,
    )

    # Match original length.
    enhanced = enhanced[: len(noisy)]

    # Prevent clipping.
    peak = np.max(np.abs(enhanced))
    if peak > 0.99:
        enhanced = enhanced * (0.99 / peak)

    return enhanced.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input noisy WAV file")
    parser.add_argument("output", help="Output denoised WAV file")
    parser.add_argument(
        "--noise-seconds",
        type=float,
        default=0.5,
        help="Duration at the beginning used for noise estimation",
    )
    args = parser.parse_args()

    noisy, sr = sf.read(args.input)

    enhanced = wiener_denoise(
        noisy,
        sr,
        noise_seconds=args.noise_seconds,
    )

    sf.write(args.output, enhanced, sr)

    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Sample rate: {sr} Hz")
    print("Wiener filtering completed.")


if __name__ == "__main__":
    main()

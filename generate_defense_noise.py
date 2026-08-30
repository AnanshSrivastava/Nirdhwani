"""
Synthesize defense-scenario noise: gunshot, rotor, artillery, vehicle engine.
No public dataset has curated versions of these at scale, so we synthesize
physically-plausible versions. Be upfront in your pitch that these are
synthesized, not field-recorded.
"""

import numpy as np
import soundfile as sf
import argparse
import os

SR = 16000  # DTLN is fixed at 16kHz, do not change


def gunshot_burst(duration=1.0, sr=SR):
    n = int(duration * sr)
    t = np.linspace(0, duration, n)
    impulse_time = np.random.uniform(0.1, 0.3)
    impulse_idx = int(impulse_time * sr)
    decay = np.exp(-40 * (t - impulse_time))
    decay[t < impulse_time] = 0
    broadband = np.random.randn(n)
    signal = broadband * decay
    signal[impulse_idx:impulse_idx + 20] += np.random.randn(20) * 3
    return signal / (np.max(np.abs(signal)) + 1e-8)


def rotor_hum(duration=1.0, sr=SR, base_freq=None):
    n = int(duration * sr)
    t = np.linspace(0, duration, n)
    f0 = base_freq or np.random.uniform(8, 20)
    signal = np.zeros(n)
    for harmonic in range(1, 6):
        signal += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
    engine_hum = 0.3 * np.sin(2 * np.pi * np.random.uniform(80, 150) * t)
    broadband = 0.2 * np.random.randn(n)
    signal = signal + engine_hum + broadband
    return signal / (np.max(np.abs(signal)) + 1e-8)


def artillery_boom(duration=1.5, sr=SR):
    n = int(duration * sr)
    t = np.linspace(0, duration, n)
    boom_time = np.random.uniform(0.1, 0.2)
    decay = np.exp(-8 * (t - boom_time))
    decay[t < boom_time] = 0
    low_rumble = np.sin(2 * np.pi * np.random.uniform(20, 60) * t)
    broadband = np.random.randn(n)
    signal = (0.6 * low_rumble + 0.4 * broadband) * decay
    return signal / (np.max(np.abs(signal)) + 1e-8)


def vehicle_engine(duration=1.0, sr=SR):
    n = int(duration * sr)
    t = np.linspace(0, duration, n)
    f0 = np.random.uniform(15, 40)
    signal = np.zeros(n)
    for harmonic in range(1, 8):
        signal += (1.0 / harmonic**1.2) * np.sin(2 * np.pi * f0 * harmonic * t)
    mechanical_noise = 0.4 * np.random.randn(n)
    signal = signal + mechanical_noise
    return signal / (np.max(np.abs(signal)) + 1e-8)


GENERATORS = {
    "gunshot": gunshot_burst,
    "rotor": rotor_hum,
    "artillery": artillery_boom,
    "vehicle": vehicle_engine,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./data/defense_noise")
    parser.add_argument("--n_per_class", type=int, default=200)
    args = parser.parse_args()

    for noise_type, gen_fn in GENERATORS.items():
        class_dir = os.path.join(args.out_dir, noise_type)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(args.n_per_class):
            duration = np.random.uniform(0.8, 2.0)
            audio = gen_fn(duration=duration)
            sf.write(os.path.join(class_dir, f"{noise_type}_{i:04d}.wav"), audio, SR)
        print(f"Generated {args.n_per_class} {noise_type} samples -> {class_dir}")


if __name__ == "__main__":
    main()
"""
Batch-process a folder of noisy .wav files through the two-stage DTLN
TF-Lite model, to verify the .tflite conversion preserved model quality
compared to the .h5 version.
"""

import soundfile as sf
import numpy as np
import tensorflow as tf
import glob
import os

block_len = 512
block_shift = 128

MODEL_1_PATH = './pretrained_model/model_1.tflite'
MODEL_2_PATH = './pretrained_model/model_2.tflite'
INPUT_DIR = './data/test_noisy'
OUTPUT_DIR = './data/test_tflite_pretrained_check'

os.makedirs(OUTPUT_DIR, exist_ok=True)

interpreter_1 = tf.lite.Interpreter(model_path=MODEL_1_PATH)
interpreter_1.allocate_tensors()
interpreter_2 = tf.lite.Interpreter(model_path=MODEL_2_PATH)
interpreter_2.allocate_tensors()

input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()
input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()

def find_inputs(input_details):
    a, b = input_details[0], input_details[1]
    if len(a['shape']) < len(b['shape']):
        return a, b
    else:
        return b, a

main_in_1, states_in_1_detail = find_inputs(input_details_1)
main_in_2, states_in_2_detail = find_inputs(input_details_2)

def find_outputs(output_details):
    a, b = output_details[0], output_details[1]
    if len(a['shape']) < len(b['shape']):
        return a, b
    else:
        return b, a

main_out_1, states_out_1_detail = find_outputs(output_details_1)
main_out_2, states_out_2_detail = find_outputs(output_details_2)

print(f"Model 1 -- main input shape: {main_in_1['shape']}, states shape: {states_in_1_detail['shape']}")
print(f"Model 2 -- main input shape: {main_in_2['shape']}, states shape: {states_in_2_detail['shape']}")


def process_file(audio):
    states_1 = np.zeros(states_in_1_detail['shape']).astype('float32')
    states_2 = np.zeros(states_in_2_detail['shape']).astype('float32')

    out_file = np.zeros((len(audio),), dtype=np.float32)
    in_buffer = np.zeros((block_len,), dtype=np.float32)
    out_buffer = np.zeros((block_len,), dtype=np.float32)
    num_blocks = max(0, (audio.shape[0] - (block_len - block_shift)) // block_shift)

    if num_blocks == 0:
        return out_file

    for idx in range(num_blocks):
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = audio[idx*block_shift:(idx*block_shift)+block_shift]

        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft)
        in_phase = np.angle(in_block_fft)
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')

        interpreter_1.set_tensor(states_in_1_detail['index'], states_1)
        interpreter_1.set_tensor(main_in_1['index'], in_mag)
        interpreter_1.invoke()
        out_mask = interpreter_1.get_tensor(main_out_1['index'])
        states_1 = interpreter_1.get_tensor(states_out_1_detail['index'])

        estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)
        estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype('float32')

        interpreter_2.set_tensor(states_in_2_detail['index'], states_2)
        interpreter_2.set_tensor(main_in_2['index'], estimated_block)
        interpreter_2.invoke()
        out_block = interpreter_2.get_tensor(main_out_2['index'])
        states_2 = interpreter_2.get_tensor(states_out_2_detail['index'])

        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer += np.squeeze(out_block)
        out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]

    return out_file


wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
print(f"Processing {len(wav_files)} files through TF-Lite model...")

for i, path in enumerate(wav_files):
    audio, fs = sf.read(path)
    if fs != 16000:
        raise ValueError(f"{path} is not 16kHz")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    out_audio = process_file(audio.astype('float32'))

    fname = os.path.basename(path)
    sf.write(os.path.join(OUTPUT_DIR, fname), out_audio, fs)

    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(wav_files)} done")

print(f"Done. Outputs in {OUTPUT_DIR}")
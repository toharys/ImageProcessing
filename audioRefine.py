import numpy as np
import matplotlib.pyplot as plt  # Optional, for visualization
from typing import Tuple
from scipy.io.wavfile import read as read_wav


def wav_to_arr(file_path) -> Tuple[np.ndarray, int]:
    frame_rate, audio_data = read_wav(file_path)

    if len(audio_data.shape) == 2:
        # If stereo, take only one channel (assuming it's the same for both)
        audio_data = audio_data[:, 0]

    return audio_data, frame_rate


def calculate_fourier_transform(audio_array, frame_rate) -> Tuple[np.ndarray, np.ndarray]:
    # Calculate the one-dimensional FFT
    fft_result = np.fft.fft(audio_array)  # Assuming mono audio, use audio_array[:, 0] for the first channel

    # Calculate the frequencies corresponding to the FFT result
    frequencies = np.fft.fftfreq(len(fft_result), d=1/frame_rate)

    # Optionally, visualize the result
    plt.plot(frequencies, np.abs(fft_result))
    plt.title('Fourier Transform of Audio')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()

    return frequencies, fft_result


def q1(audio_path) -> np.array:
    audio_arr, frame_rate = wav_to_arr(audio_path)
    frequencies, fft_result = calculate_fourier_transform(audio_arr, frame_rate)


# def q2(audio_path) -> np.array:
#     audio_arr, frame_rate = wav_to_arr(audio_path)

if __name__ == "__main__":
    q1(r"C:\Users\tohar\ImageProcessingExs\ex2\Inputs\q1.wav")
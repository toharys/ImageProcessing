import numpy as np
import matplotlib.pyplot as plt  # Optional, for visualization
from typing import Tuple
from scipy.io import wavfile
from scipy.io.wavfile import read as read_wav
from scipy.signal import iirnotch


def wav_to_arr(file_path) -> Tuple[np.ndarray, int]:
    frame_rate, audio_data = wavfile.read(file_path)
    max_int16 = np.iinfo(np.int16).max
    audio_data = (audio_data / np.max(np.abs(audio_data))) * max_int16
    return audio_data.astype(np.int16), frame_rate
    # frame_rate, audio_data = read_wav(file_path)
    #
    # if len(audio_data.shape) == 2:
    #     # If stereo, take only one channel (assuming it's the same for both)
    #     audio_data = audio_data[:, 0]
    #
    # return audio_data, frame_rate

def generate_spectrogram(audio_array, frame_rate) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    spec_values, freq_bins, time_bins, _ = plt.specgram(audio_array, Fs=frame_rate, cmap='viridis', NFFT=1024,
                                                        noverlap=512)

    # Optionally, visualize the spectrogram
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    plt.show()

    return spec_values, freq_bins, time_bins

def calculate_fourier_transform(audio_array, frame_rate) -> Tuple[np.ndarray, np.ndarray]:
    pass
#     # Calculate the one-dimensional FFT
#     fft_result = np.fft.fft(audio_array)  # Assuming mono audio, use audio_array[:, 0] for the first channel
#
#     # Calculate the frequencies corresponding to the FFT result
#     frequencies = np.fft.fftfreq(len(fft_result), d=1/frame_rate)
#
#     # Optionally, visualize the result
#     # plt.plot(frequencies, np.abs(fft_result))
#     # plt.title('Fourier Transform of Audio')
#     # plt.xlabel('Frequency (Hz)')
#     # plt.ylabel('Amplitude')
#     # plt.show()
#
#     return frequencies, fft_result

def notch_filter(audio_array, frame_rate, target_frequency, quality_factor=30):

    # Design notch filter
    nyquist = 0.5 * frame_rate
    notch_frequency = target_frequency / nyquist
    b, a = iirnotch(notch_frequency, quality_factor)

    # Apply notch filter to audio array
    filtered_audio = np.apply_along_axis(lambda x: np.convolve(x, b, mode='same'), axis=0, arr=audio_array)

    return filtered_audio

def save_audio_to_wav(audio_array, frame_rate, output_file):

    # Ensure the audio_array is in the correct data type (e.g., int16)
    audio_array = audio_array.astype(np.int16)

    # Save the audio to a WAV file
    wavfile.write(output_file, frame_rate, audio_array)

def q1(audio_path) -> np.array:
    audio_arr, frame_rate = wav_to_arr(audio_path)
    # frequencies, fft_result = calculate_fourier_transform(audio_arr, frame_rate)
    generate_spectrogram(audio_arr,frame_rate)
    filtered_audio = notch_filter(audio_arr,frame_rate,1124)
    generate_spectrogram(filtered_audio, frame_rate)
    save_audio_to_wav(filtered_audio, frame_rate, r"C:\Users\tohar\ImageProcessingExs\ex2\Outputs\q1_filtered.wav")

    # plt.plot(filtered_audio, np.abs(fft_result))
    # plt.title('Spectrogram')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # plt.colorbar(label='Intensity (dB)')
    # plt.show()

# def q2(audio_path) -> np.array:
#     audio_arr, frame_rate = wav_to_arr(audio_path)

if __name__ == "__main__":
    q1(r"C:\Users\tohar\ImageProcessingExs\ex2\Inputs\q1.wav")
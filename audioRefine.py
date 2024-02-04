import numpy as np
import matplotlib.pyplot as plt  # Optional, for visualization
from typing import Tuple
from scipy.io import wavfile
from scipy.signal import iirnotch

path_q1_denoised = r".."  # Optional, for save the denoised signal
path_q2_denoised = r".."  # Optional, for save the denoised signal


def wav_to_arr(file_path) -> Tuple[np.ndarray, int]:
    frame_rate, audio_data = wavfile.read(file_path)
    max_int16 = np.iinfo(np.int16).max
    audio_data = (audio_data / np.max(np.abs(audio_data))) * max_int16
    return audio_data.astype(np.int16), frame_rate


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


def generic_filter(audio_array, frame_rate, target_frequency, quality_factor,start_time, end_time) -> np.ndarray:

    # Design notch filter
    nyquist = 0.5 * frame_rate
    notch_frequency = target_frequency / nyquist
    b, a = iirnotch(notch_frequency, quality_factor)
    start_index = int(start_time*frame_rate)
    end_index = int(end_time*frame_rate)
    # Apply notch filter to audio array
    filtered_audio = audio_array.copy()
    filtered_audio[start_index:end_index] = np.apply_along_axis(lambda x: np.convolve(x[start_index:end_index],
        b, mode='same'), axis=0, arr=audio_array)

    return filtered_audio


def save_audio_to_wav(audio_array, frame_rate, output_file):

    # Ensure the audio_array is in the correct data type (e.g., int16)
    audio_array = audio_array.astype(np.int16)

    # Save the audio to a WAV file
    wavfile.write(output_file, frame_rate, audio_array)


def q1(audio_path) -> np.ndarray:
    audio_arr, frame_rate = wav_to_arr(audio_path)
    # frequencies, fft_result = calculate_fourier_transform(audio_arr, frame_rate)
    generate_spectrogram(audio_arr,frame_rate)
    filtered_audio = generic_filter(audio_arr, frame_rate, 1124, 4, 0.0, len(audio_arr) / frame_rate)
    generate_spectrogram(filtered_audio, frame_rate)
    save_audio_to_wav(filtered_audio, frame_rate, path_q1_denoised)
    return filtered_audio


def q2(audio_path) -> np.ndarray:
    audio_arr, frame_rate = wav_to_arr(audio_path)
    generate_spectrogram(audio_arr, frame_rate)
    filtered_audio = generic_filter(audio_arr, frame_rate, 595, 30, 1.45, 4.1)
    generate_spectrogram(filtered_audio, frame_rate)
    save_audio_to_wav(filtered_audio, frame_rate, path_q2_denoised)
    return filtered_audio



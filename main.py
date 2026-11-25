import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Sampling parameters
FS = 1000
DURATION = 2
N = FS * DURATION
CHUNK_SIZE = 50

# Time and signal buffers
time_axis = np.arange(N) / FS
signal = np.zeros(N)
sample_count = 0
phase = 0.0

# Window and frequency axis
window = np.hanning(N)
frequency_axis = np.fft.rfftfreq(N, 1 / FS)

# Create time-domain and frequency-domain plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
fig.suptitle("Real-Time Fourier Transform Visualization")

line_time, = ax1.plot(time_axis, signal, color="blue")
ax1.set_title("Time-Domain Signal")
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Amplitude")
ax1.set_ylim(-2, 2)
ax1.grid(True)

line_fft, = ax2.plot(
    frequency_axis, np.zeros(len(frequency_axis)), color="red"
)
ax2.set_title("Frequency-Domain Spectrum")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
ax2.set_xlim(0, 150)
ax2.set_ylim(0, 1.5)
ax2.grid(True)


def update(frame):
    global signal, sample_count, phase

    # Generate a slowly changing frequency
    indices = sample_count + np.arange(CHUNK_SIZE)
    t = indices / FS
    frequency = 30 + 10 * np.sin(2 * np.pi * 0.1 * t)

    # Generate a fundamental and its second harmonic
    phase_steps = 2 * np.pi * frequency / FS
    phases = phase + np.cumsum(phase_steps)
    new_signal = (
        np.sin(phases)
        + 0.5 * np.sin(2 * phases)
        + 0.05 * np.random.randn(CHUNK_SIZE)
    )

    phase = phases[-1]
    sample_count += CHUNK_SIZE

    # Move old samples out and add new samples
    signal = np.roll(signal, -CHUNK_SIZE)
    signal[-CHUNK_SIZE:] = new_signal

    # Apply window and calculate real FFT
    windowed_signal = signal * window
    fft_values = np.fft.rfft(windowed_signal)

    # Single-sided amplitude spectrum
    magnitude = 2 * np.abs(fft_values) / np.sum(window)
    magnitude[0] /= 2
    if N % 2 == 0:
        magnitude[-1] /= 2

    # Find strongest non-zero frequency bin
    peak_index = np.argmax(magnitude[1:]) + 1
    dominant_frequency = frequency_axis[peak_index]

    # Update plots
    line_time.set_ydata(signal)
    line_fft.set_ydata(magnitude)
    ax2.set_title(
        f"Frequency Spectrum - Dominant Frequency: "
        f"{dominant_frequency:.2f} Hz"
    )
    return line_time, line_fft


def main():
    animation = FuncAnimation(
        fig,
        update,
        interval=50,
        blit=False,
        cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
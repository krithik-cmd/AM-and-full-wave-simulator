import numpy as np
import matplotlib.pyplot as plt

# Simplified model:
# Diodes have a constant forward-voltage drop.
# Capacitors charge instantly when the diode conducts,
# then discharge exponentially through the load.
# Transformer resistance, charging-current peaks, and
# detailed diode characteristics are not included.


def capacitor_detector(signal, dt, resistance, capacitance, drop):
    """Diode rectification followed by capacitor smoothing."""
    output = np.zeros_like(signal)
    decay = np.exp(-dt / (resistance * capacitance))

    for i in range(1, len(signal)):
        discharged_voltage = output[i - 1] * decay
        available_voltage = max(signal[i] - drop, 0.0)
        output[i] = max(available_voltage, discharged_voltage)

    return output


def low_pass(signal, dt, cutoff):
    """First-order RC low-pass filter."""
    output = np.zeros_like(signal)
    output[0] = signal[0]
    alpha = 1 - np.exp(-2 * np.pi * cutoff * dt)

    for i in range(1, len(signal)):
        output[i] = output[i - 1] + alpha * (
            signal[i] - output[i - 1]
        )

    return output


# ============================================================
# 1. FULL-WAVE BRIDGE RECTIFIER
# ============================================================

dt = 5e-6
t = np.arange(0, 0.5, dt)

frequency = 50       # Hz
input_peak = 9       # V PEAK, matching your screenshot
load = 220          # ohms
diode_drop = 0.7     # V per diode: assumed approximation

ac_input = input_peak * np.sin(2 * np.pi * frequency * t)

# Two diodes conduct in a bridge at any instant.
rectified = np.maximum(np.abs(ac_input) - 2 * diode_drop, 0)

output_1000 = capacitor_detector(
    np.abs(ac_input), dt, load, 1000e-6, 2 * diode_drop
)

output_2200 = capacitor_detector(
    np.abs(ac_input), dt, load, 2200e-6, 2 * diode_drop
)

# Measure after startup, over five complete input cycles.
steady = t >= 0.4

print("BRIDGE RECTIFIER — SIMPLIFIED MODEL")
for name, voltage in [
    ("1000 µF", output_1000),
    ("2200 µF", output_2200),
]:
    average = np.mean(voltage[steady])
    ripple = np.ptp(voltage[steady])
    print(
        f"{name}: average = {average:.3f} V, "
        f"ripple = {ripple:.3f} V peak-to-peak"
    )

fig, axes = plt.subplots(2, 1, figsize=(11, 7))

axes[0].plot(t[steady], ac_input[steady], label="AC input")
axes[0].plot(
    t[steady], rectified[steady],
    label="Rectified output without capacitor"
)
axes[0].set_title("Full-wave bridge rectification")
axes[0].set_ylabel("Voltage (V)")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(
    t[steady], output_1000[steady], label="1000 µF"
)
axes[1].plot(
    t[steady], output_2200[steady], label="2200 µF"
)
axes[1].set_title("Effect of increasing the smoothing capacitor")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Output voltage (V)")
axes[1].legend()
axes[1].grid(True)

fig.tight_layout()
fig.savefig("bridge_rectifier.png", dpi=300)


# ============================================================
# 2. AM ENVELOPE DETECTOR
# ============================================================

# This models the selected signal AFTER the LC tuning stage.
# The antenna and LC circuit are not simulated here.

dt_am = 5e-6
t_am = np.arange(0, 0.6, dt_am)

carrier_frequency = 3000    # Hz: scaled demonstration carrier
message_frequency = 12     # Hz: scaled demonstration message
carrier_amplitude = 2      # V: chosen simulation assumption
modulation_depth = 0.5      # 50%
detector_resistance = 82e3  # 82 kΩ, matching your latest circuit
detector_capacitance = 300e-9
schottky_drop = 0.25        # V: simplified assumption, not a fixed rating

message = np.sin(2 * np.pi * message_frequency * t_am)
envelope = carrier_amplitude * (1 + modulation_depth * message)
carrier = np.sin(2 * np.pi * carrier_frequency * t_am)

clean_am = envelope * carrier

# Add controlled amplitude interference at 180 Hz.
# This represents an unwanted amplitude variation.
interference = 0.15 * np.sin(2 * np.pi * 180 * t_am)
noisy_am = (envelope + interference) * carrier

detected = capacitor_detector(
    noisy_am,
    dt_am,
    detector_resistance,
    detector_capacitance,
    schottky_drop
)

# Additional buffered RC filter:
# R = 10 kΩ, C = 330 nF -> cutoff approximately 48 Hz.
filter_r = 10e3
filter_c = 330e-9
cutoff = 1 / (2 * np.pi * filter_r * filter_c)

filtered = low_pass(detected, dt_am, cutoff)

print(f"\nAM DETECTOR: additional filter cutoff = {cutoff:.1f} Hz")
print("Carrier = 3000 Hz; message = 12 Hz")
print("AM amplitudes and diode drop are chosen model assumptions.")

view = (t_am >= 0.3) & (t_am < 0.55)

fig, axes = plt.subplots(3, 1, figsize=(11, 9))

axes[0].plot(
    t_am[view], noisy_am[view],
    linewidth=0.5, label="AM input with interference"
)
axes[0].plot(
    t_am[view], envelope[view],
    color="red", linewidth=2, label="Desired upper envelope"
)
axes[0].set_title("Selected AM signal entering the diode")
axes[0].legend()

axes[1].plot(t_am[view], detected[view], label="Detector output")
axes[1].plot(
    t_am[view], envelope[view] - schottky_drop,
    "--", label="Envelope minus assumed diode drop"
)
axes[1].set_title("Envelope detection: smoothing and tracking distortion")
axes[1].legend()

axes[2].plot(
    t_am[view], detected[view],
    alpha=0.6, label="Before additional filter"
)
axes[2].plot(
    t_am[view], filtered[view],
    linewidth=2, label="After additional filter"
)
axes[2].set_title("Additional filtering: less rapid variation, more delay")
axes[2].set_xlabel("Time (s)")
axes[2].legend()

for ax in axes:
    ax.set_ylabel("Voltage (V)")
    ax.grid(True)

fig.tight_layout()
fig.savefig("am_detector.png", dpi=300)
plt.show()
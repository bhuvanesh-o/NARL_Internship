"""
NARL 2D Plot — 40 Heights Turbulence Analysis Pipeline
Converted from Google Colab notebook for local VSCode execution.

Uses two PALM LES NetCDF files:
  - FILE_SINGLE_HEIGHT: single-height w' time series -> full timeline,
    block detrending, and the CWT input signal.
  - FILE_ALL_HEIGHTS:   multi-height (40 heights) w' data -> ensemble-
    averaged power spectrum across all heights.

Pipeline stages:
  1. Config
  2. Load single-height dataset, plot full w' time series
  3. Block detrend the single-height series (60 s blocks)
  4. Load multi-height dataset, compute ensemble-averaged power spectrum
  5. Find & annotate the dominant spectral peak
  6. Continuous Wavelet Transform (CWT) scalogram on the detrended signal
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import find_peaks
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ─────────────────────────────────────────────────────────────
# STEP 1: CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Same filenames as before — now local paths instead of Google Drive paths.
# Put these next to the script, or change to absolute paths.
FILE_SINGLE_HEIGHT = "test2m_yz.004_wprime.nc"
FILE_ALL_HEIGHTS = "test2m_yz.004_wprime (1).nc"

BLOCK_SIZE_SECONDS = 600  # 60-second blocks (approximately, per original comment)

TARGET_PERIOD_SEC = 4320.10  # 72 minutes
CWT_VIEW_MAX_HOURS = 12.0
N_SCALES = 200
DOWNSAMPLE_FACTOR = 1
DT_SEC = 1.0


# ─────────────────────────────────────────────────────────────
# STEP 2: LOAD SINGLE-HEIGHT DATASET & FULL TIME SERIES
# ─────────────────────────────────────────────────────────────

def load_single_height(file_path):
    ds = xr.open_dataset(file_path)

    print("\nDataset dimensions:")
    print(ds.sizes)

    print("\nSpatial Location:")
    print(f"X = {ds['x_yz'].values[0]} m")
    print(f"Y = {float(ds['y'].values):.2f} m")
    print(f"Z = {float(ds['zw'].values):.2f} m")

    print("\nTime information:")
    print("Start :", ds['time'][0].values)
    print("End   :", ds['time'][-1].values)
    print("Number of samples :", ds.sizes['time'])

    return ds


def extract_time_series(ds):
    time_seconds = (ds["time"].values - ds["time"].values[0]) / np.timedelta64(1, "s")
    time_hours = time_seconds / 3600.0
    w_timeline = ds["w_yz"].isel(x_yz=0).values

    print("\nLength of time array :", len(time_hours))
    print("Length of data array :", len(w_timeline))

    return time_seconds, time_hours, w_timeline


def plot_full_timeseries(ds, time_hours, w_timeline):
    plt.figure(figsize=(15, 5))

    plt.plot(time_hours, w_timeline, color="crimson", linewidth=0.6, label="w_yz")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)

    plt.title(
        f"Full Vertical Velocity Time Series\n"
        f"X={ds['x_yz'].values[0]:.1f} m, "
        f"Y={float(ds['y'].values):.1f} m, "
        f"Z={float(ds['zw'].values):.1f} m",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Simulation Time (hours)", fontsize=12)
    plt.ylabel("Vertical Velocity (m/s)", fontsize=12)
    plt.xlim(time_hours.min(), time_hours.max())
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# STEP 3: BLOCK DETRENDING
# ─────────────────────────────────────────────────────────────

def block_detrend(data, block_size=BLOCK_SIZE_SECONDS):
    out = data.copy()
    N = len(data)
    for i in range(0, N, block_size):
        end = min(i + block_size, N)
        block = data[i:end]
        out[i:end] = block - np.mean(block)
    return out


def plot_block_detrended(time_seconds, w_block, block_size=BLOCK_SIZE_SECONDS):
    time_hours = time_seconds / 3600.0

    plt.figure(figsize=(15, 5))
    plt.plot(time_hours, w_block, color='steelblue', linewidth=0.6)
    plt.axhline(0, color='black', linestyle='--')

    plt.ylabel(r"$w'$ (m/s)")
    plt.xlabel("Time (Minutes)")
    plt.title(f"{block_size}-Second Block Detrended Vertical Velocity")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("block_detrended_w.png", dpi=200)
    plt.show()


# ─────────────────────────────────────────────────────────────
# STEP 4: ENSEMBLE-AVERAGED POWER SPECTRUM ACROSS HEIGHTS
# ─────────────────────────────────────────────────────────────

def ensemble_power_spectrum(file_path, dt=1.0):
    ds = xr.open_dataset(file_path)

    n_heights = ds.sizes["zw"]
    print(f"Number of heights = {n_heights}")
    heights = ds["zw"].values

    N = ds.sizes["time"]
    freqs = np.fft.rfftfreq(N, d=dt)
    valid = freqs > 0
    freq_plot = freqs[valid]

    all_power = []

    plt.figure(figsize=(10, 7))

    for i in range(n_heights):
        z = heights[i]
        print(f"Processing height = {z:.2f} m")

        sig = ds["w_yz"].isel(zw=i, x_yz=0).values
        fft = np.fft.rfft(sig)
        power = (np.abs(fft) ** 2) / N
        all_power.append(power)

        plt.loglog(freq_plot, power[valid], color="lightgray", linewidth=0.7, alpha=0.6)

    all_power = np.array(all_power)
    power_mean = np.mean(all_power, axis=0)

    plt.loglog(freq_plot, power_mean[valid], color="red", linewidth=1, label="Mean Power Spectrum")

    plt.ylim(1e-8, 1e1)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Spectral Energy", fontsize=12)
    plt.title("Ensemble-Averaged Power Spectrum\n(Mean of All Heights)", fontsize=14, fontweight="bold")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()

    plt.tight_layout()
    plt.savefig("ensemble_power_spectrum.png", dpi=300)
    plt.show()
    print("Done!")

    return freqs, power_mean


# ─────────────────────────────────────────────────────────────
# STEP 5: FIND & PLOT DOMINANT SPECTRAL PEAK
# ─────────────────────────────────────────────────────────────

def find_and_plot_peak(freqs, power_mean):
    valid = freqs > 0
    freqs_plot = freqs[valid]
    power_plot = power_mean[valid]

    peaks, _ = find_peaks(power_plot)
    highest_peak = peaks[np.argmax(power_plot[peaks])]

    peak_freq = freqs_plot[highest_peak]
    peak_power = power_plot[highest_peak]

    print(f"Highest Peak Frequency : {peak_freq:.8f} Hz")
    print(f"Highest Peak Power     : {peak_power:.6e}")
    print(f"Corresponding Period   : {1 / peak_freq:.2f} s")
    print(f"Corresponding Period   : {1 / (60 * peak_freq):.2f} min")

    plt.figure(figsize=(10, 7))
    plt.loglog(freqs_plot, power_plot, color='firebrick', linewidth=0.8, label='Mean Power Spectrum')

    plt.scatter(peak_freq, peak_power, color='blue', s=100, zorder=5, label='Highest Peak')
    plt.annotate(
        f"{peak_freq:.3e} Hz",
        xy=(peak_freq, peak_power),
        xytext=(20, 20),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->'),
        fontsize=10
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Spectral Energy")
    plt.title("Mean Power Spectrum with Highest Peak", fontsize=13, fontweight="bold")
    plt.grid(True, which='both', linestyle=':')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return peak_freq, peak_power


# ─────────────────────────────────────────────────────────────
# STEP 6: CONTINUOUS WAVELET TRANSFORM (CWT)
# ─────────────────────────────────────────────────────────────

def build_cwt_matrix(signal_data):
    trim_len = (len(signal_data) // DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
    signal_ds = signal_data[:trim_len].reshape(-1, DOWNSAMPLE_FACTOR).mean(axis=1)
    n_samples = len(signal_ds)

    # Periods from 2 seconds to 8 hours
    period_axis = np.logspace(np.log10(2), np.log10(8 * 3600), N_SCALES)  # seconds
    scales = period_axis / (1.033043 * DT_SEC)

    cwt_power = np.zeros((N_SCALES, n_samples), dtype=np.float32)

    for i, s in enumerate(scales):
        half_width = int(5 * s)
        t = np.arange(-half_width, half_width + 1)

        norm = (1 / np.sqrt(s)) * (np.pi ** (-0.25))
        envelope = np.exp(-0.5 * (t / s) ** 2)
        oscillation = np.exp(1j * 6.0 * t / s)
        wavelet = norm * envelope * oscillation

        conv = signal.convolve(signal_ds, np.conj(wavelet), mode="same")
        cwt_power[i] = np.abs(conv) ** 2

        if (i + 1) % 50 == 0:
            print(f"Scale {i + 1}/{N_SCALES}")

    return cwt_power, signal_ds, period_axis


def run_cwt_analysis(signal_data, signal_label=r"$w'$", signal_color="firebrick"):
    print("Executing Continuous Wavelet Transform...")

    cwt_matrix, signal_ds, period_axis = build_cwt_matrix(signal_data)

    time_hours = np.arange(len(signal_ds)) * DT_SEC / 3600
    mask = time_hours <= CWT_VIEW_MAX_HOURS

    time_view = time_hours[mask]
    signal_view = signal_ds[mask]
    power_view = cwt_matrix[:, mask]

    closest_row = np.argmin(np.abs(period_axis - TARGET_PERIOD_SEC))
    energy_slice = power_view[closest_row]

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={"height_ratios": [1, 3, 1]}, sharex=True)

    # Signal
    axes[0].plot(time_view, signal_view, color=signal_color, linewidth=0.8)
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].set_ylabel(signal_label + " (m/s)")
    axes[0].set_title("Block-Detrended Vertical Velocity")
    axes[0].grid(True)

    # Scalogram
    log_power = np.log10(power_view + 1e-12)
    vmin, vmax = np.percentile(log_power, [1, 99])

    mesh = axes[1].pcolormesh(
        time_view, period_axis / 60, log_power,
        shading="auto", cmap="inferno", vmin=vmin, vmax=vmax
    )
    axes[1].axhline(TARGET_PERIOD_SEC / 60, color="lime", linestyle="--", linewidth=1.5, label="72 min")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Period (minutes)")
    axes[1].set_title("Continuous Wavelet Transform")
    axes[1].legend()

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="2%", pad=0.3)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("log10(Power)")

    # Power at target period
    axes[2].plot(time_view, energy_slice, color="lime", linewidth=1.5)
    axes[2].fill_between(time_view, 0, energy_slice, color="lime", alpha=0.25)

    mean_power = np.mean(energy_slice)
    p90_power = np.percentile(energy_slice, 90)

    axes[2].axhline(mean_power, color="red", linestyle="--", label="Mean")
    axes[2].axhline(p90_power, color="orange", linestyle=":", label="90th Percentile")
    axes[2].set_ylabel("Power")
    axes[2].set_xlabel("Time (hours)")
    axes[2].set_title("Wavelet Power at 72 Minute Period")
    axes[2].grid(True)
    axes[2].legend()
    axes[2].set_xlim(0, CWT_VIEW_MAX_HOURS)

    plt.tight_layout()
    plt.savefig("cwt_scalogram_w.png", dpi=200)
    plt.show()

    print("\nCWT Analysis Complete.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    # Single-height series: full timeline + block detrending + CWT input
    ds_single = load_single_height(FILE_SINGLE_HEIGHT)
    time_seconds, time_hours, w_timeline = extract_time_series(ds_single)
    plot_full_timeseries(ds_single, time_hours, w_timeline)

    print("\nExecuting Block Detrending on 2D Slice Temporal Data...")
    w_block = block_detrend(w_timeline)
    print("Block Detrending Complete!")
    plot_block_detrended(time_seconds, w_block)

    # Multi-height (40 heights) dataset: ensemble power spectrum
    freqs, power_mean = ensemble_power_spectrum(FILE_ALL_HEIGHTS, dt=DT_SEC)
    find_and_plot_peak(freqs, power_mean)

    # CWT on the block-detrended single-height signal
    run_cwt_analysis(w_block)


if __name__ == "__main__":
    main()
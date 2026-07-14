"""
NARL Flux Tower Data — Turbulence Analysis Pipeline
Converted from Google Colab notebook for local VSCode execution.

Pipeline stages:
  1. Config & helper functions (line sanitizing, cleaning/clipping)
  2. Data ingestion (streamed from local file instead of Google Drive)
  3. Initial cleaned time-series plots (full record + first 24h)
  4. 10-minute block detrending
  5. FFT power spectral analysis
  6. Continuous Wavelet Transform (CWT) scalograms
"""

import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

# ─────────────────────────────────────────────────────────────
# STEP 1: CONFIGURATION & SETUP
# ─────────────────────────────────────────────────────────────

# Same filename as before — now just a local path instead of a Google Drive path.
# Put the file next to this script, or change this to an absolute path.
DATA_FILE_PATH = "ts_series_data_jun_2015.txt"

COLUMNS = [
    "TIMESTAMP", "RECORD", "Ux", "Uy", "Uz", "co2", "h2o", "Ts", "press",
    "diag_csat", "t_hmp", "rh_hmp", "e_hmp", "fw", "short_up", "short_dn",
    "long_up", "long_dn", "cnr_T_K", "long_up_corr_Avg", "long_dn_corr_Avg"
]
EXPECTED_COMMAS = len(COLUMNS) - 1


def line_sanitizer(file_path, skip_rows=5):
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in range(skip_rows):
            if not f.readline():
                break

        buffer = ""
        for line in f:
            line = line.strip()
            if not line:
                continue

            combined = buffer + "," + line if buffer else line

            if combined.count(',') >= EXPECTED_COMMAS:
                yield combined + "\n"
                buffer = ""
            else:
                buffer = combined

        if buffer:
            yield buffer + "\n"


def clean_and_clip(series, var_name):
    # 1. Force to numeric
    s = pd.to_numeric(series, errors='coerce')
    s = s.replace([np.inf, -np.inf], np.nan)

    # 2. Clip outliers based on the variable
    if var_name in ['Ux', 'Uy']:
        s = s.where((s > -15) & (s < 15), np.nan)
    elif var_name == 'Uz':
        s = s.where((s > -10) & (s < 10), np.nan)

    # 3. Interpolate the gaps
    return s.interpolate(method='linear').ffill().bfill().values.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# STEP 2: DATA INGESTION (STREAMING FROM LOCAL FILE)
# ─────────────────────────────────────────────────────────────

def load_data(data_file_path):
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"❌ Error: Could not find file at: {data_file_path}")

    print(f"File verified! Streaming, repairing, and clipping data from: {data_file_path}...")

    accumulator = []
    chunk_size = 100_000
    total_rows = 0

    sanitized_stream = io.StringIO("".join(line_sanitizer(data_file_path, skip_rows=5)))

    for chunk in pd.read_csv(
        sanitized_stream,
        header=None,
        names=COLUMNS,
        chunksize=chunk_size,
        low_memory=False
    ):
        accumulator.append({
            'Ux': clean_and_clip(chunk['Ux'], 'Ux'),
            'Uy': clean_and_clip(chunk['Uy'], 'Uy'),
            'Uz': clean_and_clip(chunk['Uz'], 'Uz')
        })
        total_rows += len(chunk)
        print(f"  {total_rows:,} rows successfully processed...")

    Ux = np.concatenate([c['Ux'] for c in accumulator])
    Uy = np.concatenate([c['Uy'] for c in accumulator])
    Uz = np.concatenate([c['Uz'] for c in accumulator])
    del accumulator

    print(f"\nPhase 1 Complete! Cleaned samples ready for math: {len(Uz):,}")
    return Ux, Uy, Uz


# ─────────────────────────────────────────────────────────────
# STEP 3: INITIAL CLEANED TIME-SERIES PLOTS
# ─────────────────────────────────────────────────────────────

def plot_full_profile(Ux, Uy, Uz, stride=1):
    N = len(Uz)
    time_hours = np.arange(N) / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

    axes[0].plot(time_hours[::stride], Ux[::stride], color='navy', linewidth=0.4, alpha=0.8)
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel('$U_x$ (m/s)', fontsize=11, fontweight='bold')
    axes[0].set_title('Cleaned Time-Series Profile: Horizontal Wind Speed ($U_x$)', fontsize=12, fontweight='bold', pad=4)
    axes[0].grid(True, linestyle=':', alpha=0.5)

    axes[1].plot(time_hours[::stride], Uy[::stride], color='darkcyan', linewidth=0.4, alpha=0.8)
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel('$U_y$ (m/s)', fontsize=11, fontweight='bold')
    axes[1].set_title('Cleaned Time-Series Profile: Horizontal Wind Speed ($U_y$)', fontsize=12, fontweight='bold', pad=4)
    axes[1].grid(True, linestyle=':', alpha=0.5)

    axes[2].plot(time_hours[::stride], Uz[::stride], color='steelblue', linewidth=0.4, alpha=0.8)
    axes[2].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[2].set_ylabel('$U_z$ (m/s)', fontsize=11, fontweight='bold')
    axes[2].set_title('Cleaned Time-Series Profile: Vertical Wind Speed ($U_z$)', fontsize=12, fontweight='bold', pad=4)
    axes[2].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('sanitized_atmospheric_profile_all_vars.png', dpi=200)
    plt.show()


def plot_24h_profile(Ux, Uy, Uz, fs=1.0, stride=1):
    SAMPLES_PER_DAY = int(30 * 60 * 48 * fs)  # 86,400 samples per day

    Ux_day = Ux[:SAMPLES_PER_DAY]
    Uy_day = Uy[:SAMPLES_PER_DAY]
    Uz_day = Uz[:SAMPLES_PER_DAY]

    time_hours_day = np.arange(len(Uz_day)) / (3600.0 * fs)

    fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

    axes[0].plot(time_hours_day[::stride], Ux_day[::stride], color='navy', linewidth=0.4, alpha=0.8)
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel('$U_x$ (m/s)', fontsize=11, fontweight='bold')
    axes[0].set_title('Cleaned 24-Hour Time-Series Profile: Horizontal Wind Speed ($U_x$)', fontsize=12, fontweight='bold', pad=4)
    axes[0].grid(True, linestyle=':', alpha=0.5)

    axes[1].plot(time_hours_day[::stride], Uy_day[::stride], color='darkcyan', linewidth=0.4, alpha=0.8)
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel('$U_y$ (m/s)', fontsize=11, fontweight='bold')
    axes[1].set_title('Cleaned 24-Hour Time-Series Profile: Horizontal Wind Speed ($U_y$)', fontsize=12, fontweight='bold', pad=4)
    axes[1].grid(True, linestyle=':', alpha=0.5)

    axes[2].plot(time_hours_day[::stride], Uz_day[::stride], color='steelblue', linewidth=0.4, alpha=0.8)
    axes[2].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[2].set_ylabel('$U_z$ (m/s)', fontsize=11, fontweight='bold')
    axes[2].set_title('Cleaned 24-Hour Time-Series Profile: Vertical Wind Speed ($U_z$)', fontsize=12, fontweight='bold', pad=4)
    axes[2].grid(True, linestyle=':', alpha=0.5)

    axes[-1].set_xlabel('Timeline Interval (Hours)', fontsize=12, fontweight='bold')
    axes[-1].set_xlim(0, 24)
    plt.tight_layout()
    plt.savefig('sanitized_atmospheric_profile_24h.png', dpi=200)
    plt.show()


# ─────────────────────────────────────────────────────────────
# STEP 4: 10-MINUTE BLOCK DETRENDING
# ─────────────────────────────────────────────────────────────
# NOTE: the original notebook had two copies of this step (one sliced to
# TIME_BLOCK_HOURS, one run on the full record). The full-record version was
# the one whose output actually fed the FFT/CWT steps downstream, so that's
# the only one kept here.

BLOCK_SIZE = 10 * 60  # 10-minute blocks at 1 Hz


def block_detrend(data, block_size=BLOCK_SIZE):
    out = data.copy()
    N = len(data)
    for i in range(0, N, block_size):
        end = i + block_size
        block = data[i:end]
        out[i:end] = block - np.mean(block)
    return out


def plot_block_detrended(Ux_block, Uy_block, Uz_block, stride=1):
    N = len(Ux_block)
    time_hours = np.arange(N) / 3600

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    colors = ['royalblue', 'seagreen', 'firebrick']
    labels = ["$U_x'$ (m/s)", "$U_y'$ (m/s)", "$U_z'$ (m/s)"]
    titles = [
        "10-Minute Block Detrended Ux",
        "10-Minute Block Detrended Uy",
        "10-Minute Block Detrended Uz"
    ]
    data = [Ux_block, Uy_block, Uz_block]

    for idx in range(3):
        axes[idx].plot(time_hours[::stride], data[idx][::stride], color=colors[idx], linewidth=0.4)
        axes[idx].axhline(0, color='black', linestyle='--')
        axes[idx].set_ylabel(labels[idx])
        axes[idx].set_title(titles[idx])
        axes[idx].grid(True)

    axes[2].set_xlabel("Time (Hours)")
    plt.tight_layout()
    plt.savefig("method1_block_detrended.png", dpi=200)
    plt.show()


# ─────────────────────────────────────────────────────────────
# STEP 5: FFT POWER SPECTRAL ANALYSIS
# ─────────────────────────────────────────────────────────────

def fft_power_spectra(Ux_block, Uy_block, Uz_block, fs=1.0):
    print("Executing Step 5: FFT Power Spectral Analysis...")

    N_full = len(Ux_block)
    freqs = np.fft.rfftfreq(N_full, d=1 / fs)
    detrend_cutoff_freq = 1 / (10 * 60)

    print("FFT Length =", N_full)
    print("Frequency Resolution =", fs / N_full, "Hz")

    signals = {'Ux': Ux_block, 'Uy': Uy_block, 'Uz': Uz_block}
    colors = {'Ux': 'royalblue', 'Uy': 'seagreen', 'Uz': 'firebrick'}
    spectral_energy = {}

    for name, sig in signals.items():
        print(f"Computing FFT for {name}...")
        fft_values = np.fft.rfft(sig)
        spectral_energy[name] = (np.abs(fft_values) ** 2) / N_full

    valid_mask = freqs > 0
    freqs_plot = freqs[valid_mask]

    for name in ['Ux', 'Uy', 'Uz']:
        plt.figure(figsize=(20, 6))
        power_plot = spectral_energy[name][valid_mask]

        plt.loglog(freqs_plot, power_plot, color=colors[name], linewidth=0.7, label=f"{name}' Power Spectrum")
        plt.axvline(detrend_cutoff_freq, color='black', linestyle='--', linewidth=1.2, label='10-Min Block Boundary')

        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Spectral Energy', fontsize=12)
        plt.title(f'Power Spectrum of {name} Block-Detrended Signal', fontsize=13)
        plt.grid(True, which='both', linestyle=':')
        plt.legend()

        filename = f'fft_power_spectrum_{name}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.show()
        print(f"Saved: {filename}")

    print("Power Spectral Analysis Complete.")
    return spectral_energy, freqs


# ─────────────────────────────────────────────────────────────
# STEP 6: CONTINUOUS WAVELET TRANSFORM (CWT)
# ─────────────────────────────────────────────────────────────

TARGET_PERIODS = {'Ux': 10.0 / 60.0, 'Uy': 10.0 / 60.0, 'Uz': 10.0 / 60.0}  # hours (10 min)
CWT_VIEW_MAX_HOURS = 10
N_SCALES = 200
DOWNSAMPLE_FACTOR = 5
DT_SEC = 5.0


def build_cwt_matrix(signal_data):
    trim_len = (len(signal_data) // DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
    signal_ds = signal_data[:trim_len].reshape(-1, DOWNSAMPLE_FACTOR).mean(axis=1)
    n_samples = len(signal_ds)

    # Period axis: 10 s up to 2 hours
    period_axis = np.logspace(np.log10(10 / 3600), np.log10(2.0), N_SCALES)
    scales = (period_axis * 3600) / (1.033043 * DT_SEC)

    cwt_power = np.zeros((N_SCALES, n_samples), dtype=np.float32)

    for i, s in enumerate(scales):
        half_width = int(5 * s)
        t = np.arange(-half_width, half_width + 1)

        norm = (1 / np.sqrt(s)) * (np.pi ** (-0.25))
        envelope = np.exp(-0.5 * (t / s) ** 2)
        oscillation = np.exp(1j * 6.0 * t / s)
        wavelet = norm * envelope * oscillation

        conv = signal.convolve(signal_ds, np.conj(wavelet), mode='same')
        cwt_power[i] = np.abs(conv) ** 2

        if (i + 1) % 50 == 0:
            print(f"Scale {i + 1}/{N_SCALES}")

    return cwt_power, signal_ds, period_axis


def run_cwt_analysis(Ux_block, Uy_block, Uz_block):
    print("Executing Continuous Wavelet Transform...")

    components = {'Ux': Ux_block, 'Uy': Uy_block, 'Uz': Uz_block}
    colors = {'Ux': 'royalblue', 'Uy': 'seagreen', 'Uz': 'firebrick'}

    for name, data in components.items():
        print(f"\nProcessing {name}...")

        cwt_matrix, signal_ds, period_axis = build_cwt_matrix(data)

        time_hours = (np.arange(len(signal_ds)) * DT_SEC) / 3600
        mask = time_hours <= CWT_VIEW_MAX_HOURS

        time_view = time_hours[mask]
        signal_view = signal_ds[mask]
        power_view = cwt_matrix[:, mask]

        target_period = TARGET_PERIODS[name]
        closest_row = np.argmin(np.abs(period_axis - target_period))
        energy_slice = power_view[closest_row]

        fig, axes = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [1, 3, 1]}, sharex=True)

        axes[0].plot(time_view, signal_view, color=colors[name], linewidth=0.8)
        axes[0].axhline(0, color='black', linestyle='--')
        axes[0].set_ylabel(f"{name}' (m/s)")
        axes[0].set_title(f"{name} Block-Detrended Signal")
        axes[0].grid(True)

        log_power = np.log10(power_view + 1e-12)
        vmin, vmax = np.percentile(log_power, [1, 99])

        mesh = axes[1].pcolormesh(time_view, period_axis * 60, log_power, shading='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        axes[1].axhline(period_axis[closest_row] * 60, color='lime', linestyle='--', linewidth=1.5, label=f'{period_axis[closest_row] * 60:.1f} min')
        axes[1].set_yscale('log')
        axes[1].set_ylabel('Period (Minutes)')
        axes[1].set_title(f'{name} Continuous Wavelet Transform')
        axes[1].legend()

        cbar = fig.colorbar(mesh, ax=axes[1])
        cbar.set_label('log10(Power)')

        axes[2].plot(time_view, energy_slice, color='lime', linewidth=1.5)
        axes[2].fill_between(time_view, 0, energy_slice, color='lime', alpha=0.2)

        mean_power = np.mean(energy_slice)
        p90_power = np.percentile(energy_slice, 90)

        axes[2].axhline(mean_power, color='red', linestyle='--', label='Mean')
        axes[2].axhline(p90_power, color='orange', linestyle=':', label='90th Percentile')
        axes[2].set_ylabel('Power')
        axes[2].set_xlabel('Time (Hours)')
        axes[2].set_title('Wavelet Power at Target Scale')
        axes[2].grid(True)
        axes[2].legend()

        plt.xlim(0, CWT_VIEW_MAX_HOURS)
        plt.tight_layout()

        filename = f'cwt_scalogram_{name}.png'
        plt.savefig(filename, dpi=200)
        plt.show()
        print(f"Saved: {filename}")

    print("\nCWT Analysis Complete.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    fs = 1.0

    Ux, Uy, Uz = load_data(DATA_FILE_PATH)

    plot_full_profile(Ux, Uy, Uz)
    plot_24h_profile(Ux, Uy, Uz, fs=fs)

    print("\nExecuting Block Detrending...")
    Ux_block = block_detrend(Ux)
    Uy_block = block_detrend(Uy)
    Uz_block = block_detrend(Uz)
    print("Block Detrending Complete!")

    plot_block_detrended(Ux_block, Uy_block, Uz_block)

    fft_power_spectra(Ux_block, Uy_block, Uz_block, fs=fs)

    run_cwt_analysis(Ux_block, Uy_block, Uz_block)


if __name__ == "__main__":
    main()
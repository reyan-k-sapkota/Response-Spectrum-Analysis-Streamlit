# Magnitude lookup 
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from matplotlib.ticker import MultipleLocator, AutoMinorLocator



EARTHQUAKE_MAGNITUDES = {
    "cmendocino":            {"magnitude": 7.1, "year": 1992, "name": "Cape Mendocino Earthquake",                "dt": 0.02},
    "erzincan":              {"magnitude": 6.9, "year": 1992, "name": "Erzincan Earthquake",                      "dt": 0.0025},
    "gorkha":                {"magnitude": 7.8, "year": 2015, "name": "Gorkha Earthquake",                        "dt": 0.005},
    "kobe":                  {"magnitude": 6.9, "year": 1995, "name": "Kobe Earthquake",                          "dt": 0.01},
    "landers":               {"magnitude": 7.3, "year": 1992, "name": "Landers Earthquake",                       "dt": 0.002},
    "lomaprieta_lexdam":     {"magnitude": 6.9, "year": 1989, "name": "Loma Prieta (Lexington Dam) Earthquake",   "dt": 0.01},
    "lomaprieta_losgatos":   {"magnitude": 6.9, "year": 1989, "name": "Loma Prieta (Los Gatos) Earthquake",       "dt": 0.005},
    "northridge_oliveview":  {"magnitude": 6.7, "year": 1994, "name": "Northridge (Olive View) Earthquake",       "dt": 0.01},
    "northridge_rinaldi":    {"magnitude": 6.7, "year": 1994, "name": "Northridge (Rinaldi) Earthquake",          "dt": 0.0025},
    "tabas":                 {"magnitude": 7.4, "year": 1978, "name": "Tabas Earthquake",                         "dt": 0.01},
}

def get_earthquake_info(filename):
    """Extract earthquake key from filename and return its info."""
    base = filename.replace(".txt", "").strip()
    for key in EARTHQUAKE_MAGNITUDES:
        if base.lower().startswith(key):
            return EARTHQUAKE_MAGNITUDES[key]
    return None



def acceleration_vs_time(time, accelerations, max_t, max_val, min_t, min_val, eq_name):

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    ax.plot(time, accelerations, color='#00c8ff', linewidth=0.6, alpha=0.9, label='Ground Acceleration')

    ax.fill_between(time, accelerations, 0,
        where=[a > 0 for a in accelerations],
        color='#00c8ff', alpha=0.15)
    ax.fill_between(time, accelerations, 0,
        where=[a < 0 for a in accelerations],
        color='#ff4b6e', alpha=0.15)

    ax.axvline(x=max_t, color='#f9c74f', linewidth=1, linestyle='--', alpha=0.7)
    ax.scatter([max_t], [max_val], color='#f9c74f', zorder=5, s=60)
    ax.annotate(f'Max +ve acceleration\n  {max_val:.5f}\n  t = {max_t:.3f}s',
        xy=(max_t, max_val), fontsize=8.5, color='#f9c74f',
        xytext=(max_t + 1, max_val * 0.85),
        arrowprops=dict(arrowstyle='->', color='#f9c74f', lw=1.2))

    ax.axvline(x=min_t, color='#ff4b6e', linewidth=1, linestyle='--', alpha=0.7)
    ax.scatter([min_t], [min_val], color='#ff4b6e', zorder=5, s=60)
    ax.annotate(f'  Max -ve acceleration\n  {min_val:.5f}\n  t = {min_t:.3f}s',
        xy=(min_t, min_val), fontsize=8.5, color='#ff4b6e',
        xytext=(min_t + 1, min_val * 0.75),
        arrowprops=dict(arrowstyle='->', color='#ff4b6e', lw=1.2))

    ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Time (s)', color='white', fontsize=11)
    ax.set_ylabel('Acceleration (g)', color='white', fontsize=11)
    ax.set_title(f'{eq_name} — Ground Acceleration Time History',
        color='white', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.set_xlim(0, max(time))
    ax.grid(True, color='#333', linewidth=0.5, alpha=0.7)
    ax.legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white',
        fontsize=9, loc='upper right')

    plt.tight_layout()
    return fig


def fft (DT, accelerations, eq_name):
    acc = np.array(accelerations)
    n = len(acc)
    
    # FFT
    fft_vals = np.fft.fft(acc)
    
    # frequencies corresponding to FFT bins
    freq = np.fft.fftfreq(n, DT)
     
    # amplitude spectrum
    amplitude = np.abs(fft_vals) / n
    
    # keep only positive frequencies
    mask = freq >= 0
    
    freq_pos = freq[mask]
    amp_pos = amplitude[mask]
    
    # ignore DC component (0 Hz)
    freq_no_dc = freq_pos[1:]
    amp_no_dc = amp_pos[1:]
    
    # dominant frequency
    dominant_idx = np.argmax(amp_no_dc)
    
    dominant_freq = freq_no_dc[dominant_idx]
    dominant_amp = amp_no_dc[dominant_idx]
    
    # Plot
    
    fig, ax = plt.subplots(figsize=(14,5))
    
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')
    
    # FFT curve
    
    ax.plot(
    freq_pos,
    amp_pos,
    color='#00c8ff',
    linewidth=0.8,
    alpha=0.95,
    label='FFT Amplitude Spectrum'
    )
    
    # dominant frequency marker
    ax.axvline(
    dominant_freq,
    color='#f9c74f',
    linestyle='--',
    linewidth=1.2,
    alpha=0.8
    )
    
    ax.scatter(
    dominant_freq,
    dominant_amp,
    color='#f9c74f',
    s=65,
    zorder=5
    )
    
    # annotation
    ax.annotate(
    f'Dominant Frequency\n{dominant_freq:.3f} Hz',
    xy=(dominant_freq, dominant_amp),
    xytext=(dominant_freq + 0.6, dominant_amp*0.85),
    fontsize=9,
    color='#f9c74f',
    arrowprops=dict(
        arrowstyle='->',
        color='#f9c74f',
        lw=1.2
    )
    )
    
    # styling
    ax.set_xlim(0, 10)
    
    ax.set_xlabel(
    'Frequency (Hz)',
    color='white',
    fontsize=11
    )
    
    ax.set_ylabel(
    'Amplitude',
    color='white',
    fontsize=11
    )
    
    ax.set_title(
    f'{eq_name} — Fast Fourier Transformation',
    color='white',
    fontsize=13,
    fontweight='bold',
    pad=12
    )
    
    ax.tick_params(colors='white')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
        
    ax.grid(
        True,
        color='#333',
        linewidth=0.5,
        alpha=0.7
    )
    
    ax.legend(
    facecolor='#1a1a2e',
    edgecolor='#444',
    labelcolor='white',
    fontsize=9,
    loc='upper right'
    )
    
    plt.tight_layout()
    return fig, dominant_freq

@njit
def newmark_response(accelerations, periods, zeta, m, DT, beta, gamma):

    # convert g → m/s²
    accelerations = accelerations * 9.81

    n_periods = len(periods)
    n = len(accelerations)

    Sd = np.zeros(n_periods)   # cm
    Sv = np.zeros(n_periods)   # cm/s
    Sa = np.zeros(n_periods)   # g

    for j in range(n_periods):

        Tn = periods[j]

        wn = 2*np.pi / Tn
        k = m * wn**2
        c = 2*zeta*m*wn

        u = np.zeros(n)
        v = np.zeros(n)
        a = np.zeros(n)

        a[0] = (
            -m*accelerations[0]
            - c*v[0]
            - k*u[0]
        ) / m

        khat = (
            k
            + (gamma/(beta*DT))*c
            + m/(beta*DT**2)
        )

        for i in range(n-1):

            p_eff = (
                -m*accelerations[i+1]

                + m*(
                    u[i]/(beta*DT**2)
                    + v[i]/(beta*DT)
                    + (1/(2*beta)-1)*a[i]
                )

                + c*(
                    (gamma/(beta*DT))*u[i]
                    + (gamma/beta-1)*v[i]
                    + DT*(gamma/(2*beta)-1)*a[i]
                )
            )

            u[i+1] = p_eff / khat

            a[i+1] = (
                (u[i+1]-u[i])/(beta*DT**2)
                - v[i]/(beta*DT)
                - (1/(2*beta)-1)*a[i]
            )

            v[i+1] = (
                v[i]
                + DT*((1-gamma)*a[i] + gamma*a[i+1])
            )

        # absolute acceleration
        a_abs = a + accelerations

        # convert outputs
        Sd[j] = np.max(np.abs(u)) * 100        # m → cm
        Sv[j] = np.max(np.abs(v)) * 100        # m/s → cm/s
        Sa[j] = np.max(np.abs(a_abs)) / 9.81  # m/s² → g

    return Sd, Sv, Sa


def generate_RSA_SpectralValues(S1, S2, S3, damping_ratios, periods, eq_name, type, type_unit):
    bg     = "#0b1020"
    c_grid = "#2a2a3a"
    txt    = "#f5f5f5"

    # Three distinct curve colors
    colors = ["#00c8ff", "#f9c74f", "#ff4b6e"]
    labels = [
        f"ζ = {damping_ratios[0]*100:.1f}%",
        f"ζ = {damping_ratios[1]*100:.1f}%",
        f"ζ = {damping_ratios[2]*100:.1f}%",
    ]

    def style_ax(ax, ylabel, title):
        ax.set_facecolor(bg)
        ax.set_xlabel("Natural Time Period, T (s)", color=txt, fontsize=10)
        ax.set_ylabel(ylabel, color=txt, fontsize=10)
        ax.set_title(title, color=txt, fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(axis="both", colors=txt, labelsize=9,
            which="both", direction="in", length=4)
        ax.tick_params(axis="both", which="minor", length=2)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444455")
        ax.grid(True, which="major", color=c_grid, linewidth=0.6, alpha=0.9)
        ax.grid(True, which="minor", color=c_grid, linewidth=0.3, alpha=0.5, linestyle=":")
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_xlim(periods[0], periods[-1])

    fig_s, ax1 = plt.subplots(figsize=(14, 5))
    fig_s.patch.set_facecolor(bg)

    # Plot all three curves
    for Sd, color, label in zip([S1, S2, S3], colors, labels):
        ax1.plot(periods, Sd, color=color, linewidth=0.9, alpha=0.9, label=label)

    style_ax(ax1,
        ylabel=f"Spectral {type},  $S_d$  ({type_unit})",
        title=f"Spectral {type} Response Spectrum for {eq_name}"
    )

    ax1.legend(
        facecolor="#1a1a2e", edgecolor="#444455",
        labelcolor=txt, fontsize=10,
        loc="upper right"
    )

    fig_s.tight_layout()
    return fig_s

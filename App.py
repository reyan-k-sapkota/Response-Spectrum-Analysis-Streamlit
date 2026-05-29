import streamlit as st
import os
import matplotlib.pyplot as plt
import io
import numpy as np
from HelperFunctions import get_earthquake_info, acceleration_vs_time, fft, newmark_response, generate_RSA_SpectralValues

st.set_page_config(
    page_title="Response Spectrum Curve",
    page_icon="📈",
    layout="wide"
)


st.title("📈 Response Spectrum Analysis")

st.markdown("""
<h4 style="color: #e2e8f0;">Welcome to the Web Application version of
</h4>
""", unsafe_allow_html=True)

st.markdown("""
<h4 style="color: #e2e8f0;">
<strong>Assignment 1: Response Spectrum Analysis of Earthquake Ground Motion Records</strong></h4>
""", unsafe_allow_html=True)


st.markdown("""
1. Generate **Ground Acceleration Time History** from the selected earthquake record.
2. Conduct **Fast Fourier Transform (FFT)** analysis to identify dominant frequencies.
3. Calculate values of **Spectral Displacements, Spectral Velocities, and Spectral Accelerations** for a range of natural periods by solving SDOF equation of motion using the **Newmark Method**.
4. Generate and plot the **Response Spectrum Curve** based on the calculated values.
""")

st.space()
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**Submitted to:**  
Kshitiz Charan Shrestha sir 
Associate Professor  
Department of Civil Engineering  
Institute of Engineering, Pulchowk Campus.
""")

with col2:
    st.markdown("""
**Submitted by:**  
Reyan Kumar Sapkota  
079BCE137  
BE, Civil Engineering.  
Institute of Engineering, Pulchowk Campus.
""")
    
st.markdown("""
<p style="font-size: 13px;"><i>Seismic Resistant Design of Masonry Structures</i></p>
""", unsafe_allow_html=True)
st.divider()

##############################################################################
##############################################################################
##############################################################################

# Earthquake Record Selection
st.markdown("""
<h2 style="color: #e2e8f0;"><strong> 1. Select Earthquake Record</strong></h2>
""", unsafe_allow_html=True)

RECORDS_FOLDER = "Earthquake Records"
txt_files = [f for f in os.listdir(RECORDS_FOLDER) if f.endswith(".txt")]
selected_record = st.selectbox(
    "Choose an earthquake record:",
    options=txt_files,
    index=0,
)


file_path = os.path.join(RECORDS_FOLDER, selected_record)
eq_info   = get_earthquake_info(selected_record)

# Auto-read DT from dictionary
DT = eq_info["dt"]

st.info(f"Time Interval: **DT = {DT} s** (from record metadata)")

with open(file_path, "r") as f:
    data = f.read().splitlines()

accelerations = np.array([float(val.strip()) for val in data if val.strip()])
time          = [i * DT for i in range(len(accelerations))]
max_val       = max(accelerations)
min_val       = min(accelerations)
max_t         = accelerations.tolist().index(max_val) * DT
min_t         = accelerations.tolist().index(min_val) * DT

st.session_state["time"]          = time
st.session_state["accelerations"] = accelerations
st.session_state["max_val"]       = max_val
st.session_state["min_val"]       = min_val
st.session_state["max_t"]         = max_t
st.session_state["min_t"]         = min_t
st.session_state["DT"]            = DT

if eq_info:
    st.success(f"**{eq_info['name']} ({eq_info['year']})** — Mw {eq_info['magnitude']}")
else:
    st.success(f"Selected: **{selected_record}**")

if st.button("Display Earthquake Record Summary", use_container_width=True):
    st.subheader("Record Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Magnitude (Mw)",       f"{eq_info['magnitude']}" if eq_info else "N/A")
    c2.metric("Total Data Points",    f"{len(accelerations):,}")
    c3.metric("Duration",             f"{max(time):.3f} s")
    c4.metric("Max +ve Acceleration", f"{max_val:.6f} g", f"At time, t = {max_t:.3f} s")
    c5.metric("Max -ve Acceleration", f"{min_val:.6f} g", f"At time, t = {min_t:.3f} s")



st.divider()

st.markdown("""
<h2 style="color: #e2e8f0;"><strong>2. Ground Acceleration Time History curve</strong></h2>
""", unsafe_allow_html=True)

#Acceleration vs Time 
if st.button("Generate Acceleration vs Time", use_container_width=True):
    with st.spinner("Generating Acceleration vs Time History..."):
        pga   = max(abs(max_val), abs(min_val))
        pga_t = max_t if abs(max_val) >= abs(min_val) else min_t
        
        eq_name = eq_info["name"] if eq_info else selected_record
        fig = acceleration_vs_time(time, accelerations, max_t, max_val, min_t, min_val, eq_name)
        
        # Save to buffer and store in session_state
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=450, bbox_inches='tight', facecolor='#0f1117')
        buf.seek(0)
        plt.close(fig)
        
        st.session_state["fig_acc_buf"] = buf
        st.session_state["pga"]         = pga
        st.session_state["pga_t"]       = pga_t

# Render OUTSIDE button block — persists across reruns
if "fig_acc_buf" in st.session_state:
    st.image(st.session_state["fig_acc_buf"])
    st.metric("PGA", f"{st.session_state['pga']:.6f} g", f"At time, t = {st.session_state['pga_t']:.3f} s")
    st.download_button(
        label="⬇️ Download Plot as PNG",
        data=st.session_state["fig_acc_buf"],
        file_name=f"{selected_record.replace('.txt', '')}_acc_time_history.png",
        mime="image/png",
        key = "AccVsTime"
    )

st.divider()

st.markdown("""
<h2 style="color: #e2e8f0;"><strong>3. Fast Fourier Transform</strong></h2>
""", unsafe_allow_html=True)

# FFT 
if st.button("Dominant Frequency & Amplitude using FFT", use_container_width=True):
    with st.spinner("Generating Amplitude vs Frequency Curve through Fast Fourier Transform..."):
        eq_name = eq_info["name"] if eq_info else selected_record
        fig1 = fft(DT, accelerations, eq_name)[0]
        dominant_freq = fft(DT, accelerations, eq_name)[1]
        
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png", dpi=450, bbox_inches='tight', facecolor='#0f1117')
        buf1.seek(0)
        plt.close(fig1)
        
        st.session_state["fig_fft_buf"]      = buf1
        st.session_state["dominant_freq"]    = dominant_freq

if "fig_fft_buf" in st.session_state:
    st.image(st.session_state["fig_fft_buf"])
    st.metric("Dominant Frequency", f"{st.session_state['dominant_freq']:.6f} Hz")
    st.download_button(
        label="⬇️ Download Plot as PNG",
        data=st.session_state["fig_fft_buf"],
        file_name=f"{selected_record.replace('.txt', '')}_FFT.png",
        mime="image/png",
        key = "fft"
    )

st.divider()

st.markdown("""
<h2 style="color: #e2e8f0;"><strong>4. Spectral Values from Newmark Method</strong></h2>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Newmark Parameter (β)", "1/4")
col2.metric("Newmark Parameter (γ)", "1/2")
col3.metric("Mass (m)", "1 kg")

#st.subheader("Input the values of three damping ratios (in %)")
st.divider()

st.markdown("""
<p style="text-align: center; font-size: 1.2em; font-weight: 600;">
Input the values of three damping ratios (in %)
</p>
""", unsafe_allow_html=True)


d1, d2, d3 = st.columns(3)

with d1:
    zeta1 = st.number_input("Damping Ratio (ζ1)", min_value=0.0, max_value=100.0, value=2.0, step=0.5, format="%.1f")

with d2:
    zeta2 = st.number_input("Damping Ratio (ζ2)", min_value=0.0, max_value=100.0, value=5.0, step=0.5, format="%.1f")

with d3:
    zeta3 = st.number_input("Damping Ratio (ζ3)", min_value=0.0, max_value=100.0, value=10.0, step=0.5, format="%.1f")

damping_ratios = [zeta1 / 100, zeta2 / 100, zeta3 / 100]

st.divider()

st.markdown("""
<p style="text-align: center; font-size: 1.2em; font-weight: 600;">
Input the Natural Time period Range for Response Spectrum Curves
</p>
""", unsafe_allow_html=True)

p1, p2, p3 = st.columns(3)

with p1:
    T_start = st.number_input(
        "Starting Period (s)",
        min_value=0.001,
        max_value=10.0,
        value=0.05,
        step=0.1,
        format="%.3f",
    )

with p2:
    T_end = st.number_input(
        "End Period (s)",
        min_value=0.001,
        max_value=15.0,
        value=5.05,
        step=0.1,
        format="%.3f",
    )

with p3:
    T_interval = st.selectbox(
        "Period Interval (s)",
        options=[0.001, 0.01, 0.1, 0.005],
        index=1,                      # default: 0.01
        format_func=lambda x: str(x), # clean display
    )

# Generate period array for calculations
periods = np.arange(T_start, T_end + T_interval, T_interval)
st.caption(f"Total periods to evaluate: **{len(periods):,}**  |  Range: {T_start}s → {T_end}s  |  Interval: {T_interval}s")

st.divider()

# --- Single Calculate Button ---
if st.button("Run Newmark Iterations for calculating tripartite spectral values", use_container_width=True, type="primary"):

    m     = 1
    beta  = 0.25
    gamma = 0.5

    zeta1 = damping_ratios[0]
    zeta2 = damping_ratios[1]
    zeta3 = damping_ratios[2]

    acc = np.array(st.session_state["accelerations"], dtype=np.float64)
    DT  = st.session_state["DT"]

    with st.spinner("Running Newmark Method for 3 damping ratios... this may take a moment."):
        Sd1, Sv1, Sa1 = newmark_response(acc, periods, zeta1, m, DT, beta, gamma)
        Sd2, Sv2, Sa2 = newmark_response(acc, periods, zeta2, m, DT, beta, gamma)
        Sd3, Sv3, Sa3 = newmark_response(acc, periods, zeta3, m, DT, beta, gamma)

    # Store everything in session_state
    st.session_state["Sd1"]            = Sd1
    st.session_state["Sd2"]            = Sd2
    st.session_state["Sd3"]            = Sd3
    st.session_state["Sv1"]            = Sv1
    st.session_state["Sv2"]            = Sv2
    st.session_state["Sv3"]            = Sv3
    st.session_state["Sa1"]            = Sa1
    st.session_state["Sa2"]            = Sa2
    st.session_state["Sa3"]            = Sa3
    st.session_state["periods"]        = periods
    st.session_state["damping_ratios"] = damping_ratios

    st.success("Calculations completed!")


# Show summary table only after calculations are done
if "Sd1" in st.session_state:
    dr = st.session_state["damping_ratios"]
    p  = st.session_state["periods"]
    st.markdown("**Peak Values Summarised**")
    c1, c2, c3 = st.columns(3)
    for col, Sd, Sv, Sa, z in zip(
        [c1, c2, c3],
        [st.session_state["Sd1"], st.session_state["Sd2"], st.session_state["Sd3"]],
        [st.session_state["Sv1"], st.session_state["Sv2"], st.session_state["Sv3"]],
        [st.session_state["Sa1"], st.session_state["Sa2"], st.session_state["Sa3"]],
        dr
    ):
        col.metric(f"ζ = {z*100:.1f}%  |  Peak Sd", f"{max(Sd):.4f} cm")
        col.metric(f"ζ = {z*100:.1f}%  |  Peak Sv", f"{max(Sv):.4f} cm/s")
        col.metric(f"ζ = {z*100:.1f}%  |  Peak Sa", f"{max(Sa):.4f} g")

st.divider()

st.markdown("""
<h2 style="color: #e2e8f0;"><strong>🌐5. Response Spectrum Curves</strong></h2>
""", unsafe_allow_html=True)

#Generate RS plots
if st.button("Generate Response Spectrum for Displacement", use_container_width=True):

    with st.spinner("Generating RS Curve for Spectral Displacement vs Natural Time Periods..."):
        eq_name = eq_info["name"] if eq_info else selected_record
        Sd1  = st.session_state["Sd1"]  
        Sd2= st.session_state["Sd2"]
        Sd3= st.session_state["Sd3"]
        damping_ratios = st.session_state["damping_ratios"]
        
        fig_sd  = generate_RSA_SpectralValues(Sd1, Sd2, Sd3, damping_ratios, periods, eq_name, type = "Displacement", type_unit = "cm")
        
        buf_sd = io.BytesIO()
        fig_sd.savefig(buf_sd, format="png", dpi=450, bbox_inches='tight', facecolor='#0f1117')
        buf_sd.seek(0)
        
        plt.close(fig_sd)
        st.session_state["fig_sd_buf"] = buf_sd

# Render OUTSIDE button block — persists across reruns
if "fig_sd_buf" in st.session_state:
    st.image(st.session_state["fig_sd_buf"])
    st.download_button(
        label="⬇️ Download Plot as PNG",
        data=st.session_state["fig_sd_buf"],
        file_name=f"{selected_record.replace('.txt', '')}_SD_spectrum.png",
        mime="image/png",
        key = "sd"
    )

st.divider()

if st.button("Generate Response Spectrum for Velocity", use_container_width=True):

    with st.spinner("Generating RS curve for Spectral Acceleration vs Natural Time Periods..."):
        eq_name = eq_info["name"] if eq_info else selected_record
        Sv1  = st.session_state["Sv1"]  
        Sv2= st.session_state["Sv2"]
        Sv3= st.session_state["Sv3"]
        damping_ratios = st.session_state["damping_ratios"]
        
        fig_sv  = generate_RSA_SpectralValues(Sv1, Sv2, Sv3, damping_ratios, periods, eq_name, type = "Velocity", type_unit = "cm/s")
        
        buf_sv = io.BytesIO()
        fig_sv.savefig(buf_sv, format="png", dpi=450, bbox_inches='tight', facecolor='#0f1117')
        buf_sv.seek(0)
        
        plt.close(fig_sv)
        st.session_state["fig_sv_buf"] = buf_sv

# Render OUTSIDE button block — persists across reruns
if "fig_sv_buf" in st.session_state:
    st.image(st.session_state["fig_sv_buf"])
    st.download_button(
        label="⬇️ Download Plot as PNG",
        data=st.session_state["fig_sv_buf"],
        file_name=f"{selected_record.replace('.txt', '')}_SV_spectrum.png",
        mime="image/png",
        key = "sv"
    )

st.divider()


if st.button("Generate Response Spectrum for Acceleration", use_container_width=True):

    with st.spinner("Generating RS curve for Spectral Acceleration vs Natural Time Periods..."):
        eq_name = eq_info["name"] if eq_info else selected_record
        Sa1  = st.session_state["Sa1"]  
        Sa2= st.session_state["Sa2"]
        Sa3= st.session_state["Sa3"]
        damping_ratios = st.session_state["damping_ratios"]
        
        fig_sa  = generate_RSA_SpectralValues(Sa1, Sa2, Sa3, damping_ratios, periods, eq_name, type = "Acceleration", type_unit = "g")
        
        buf_sa = io.BytesIO()
        fig_sa.savefig(buf_sa, format="png", dpi=450, bbox_inches='tight', facecolor='#0f1117')
        buf_sa.seek(0)
        
        plt.close(fig_sa)
        st.session_state["fig_sa_buf"] = buf_sa

# Render OUTSIDE button block — persists across reruns
if "fig_sa_buf" in st.session_state:
    st.image(st.session_state["fig_sa_buf"])
    st.download_button(
        label="⬇️ Download Plot as PNG",
        data=st.session_state["fig_sa_buf"],
        file_name=f"{selected_record.replace('.txt', '')}_SA_spectrum.png",
        mime="image/png",
        key = "sa"
    )

st.divider()

#TradeMmarks


st.markdown("""
<style>

/* remove Streamlit bottom padding */
.block-container {
    padding-bottom: 0rem;
}

/* footer */
.footer {
    left: 0;
    bottom: 0;
    width: 100%;
    margin: 0;
    padding: 6px 0;
    background: #0f1117;
    border-top: 0.5px solid #2a2f3a;

    text-align: center;
    color: #9aa4b2;
    font-size: 13px;
    letter-spacing: 0.3px;

    z-index: 999;
}
</style>

<div class="footer">
    📈 Response Spectrum Analysis<br>
    Created by <b>  <a href="https://www.linkedin.com/in/reyan-k-sapkota/" target="_blank">Reyan Kumar Sapkota</a></b> |Institute of Engineering, Pulchowk Campus<br>
    © 2026 
</div>
""", unsafe_allow_html=True)
import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="EE Circuit Designer", layout="wide")
st.title("EE Circuit Design Tool")
st.caption("Phase 3 · Filter Calculation Engine")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filter Parameters")
    filter_type = st.selectbox("Type", ["Low-pass", "High-pass", "Band-pass", "Band-stop"])
    approx      = st.selectbox("Approximation",
                               ["Butterworth", "Chebyshev I", "Chebyshev II", "Elliptic", "Bessel"])
    order = st.slider("Order", 1, 8, 2)

    is_band = filter_type in ["Band-pass", "Band-stop"]

    if is_band:
        st.markdown("**Frequency Band**")
        ca, cb = st.columns(2)
        f_low  = ca.number_input("f_low (Hz)",  1.0, 1e5,  800.0, step=10.0)
        f_high = cb.number_input("f_high (Hz)", 1.0, 1e5, 1200.0, step=10.0)
        if f_high <= f_low:
            st.error("f_high must be greater than f_low")
            st.stop()
        f_center = float(np.sqrt(f_low * f_high))
        bw       = f_high - f_low
        Q_band   = f_center / bw
        st.caption(f"f₀ = {f_center:.1f} Hz · BW = {bw:.1f} Hz · Q = {Q_band:.2f}")
        fc = f_center
    else:
        fc = st.number_input("Cutoff Frequency (Hz)", 1.0, 1e5, 1000.0, step=100.0)

    impedance   = st.number_input("Base Impedance R (Ω)", 100.0, 1e6, 10000.0, step=1000.0)
    ripple_pass = 0.5
    ripple_stop = 40.0
    if approx in ["Chebyshev I", "Elliptic"]:
        ripple_pass = st.slider("Passband Ripple (dB)", 0.1, 3.0, 0.5, step=0.1)
    if approx in ["Chebyshev II", "Elliptic"]:
        ripple_stop = st.slider("Stopband Attenuation (dB)", 20.0, 80.0, 40.0, step=1.0)

# ── SI prefix formatter ───────────────────────────────────────────────────────
def fmt(val, unit):
    for factor, prefix in [(1e12,"T"),(1e9,"G"),(1e6,"M"),(1e3,"k"),
                            (1,""),(1e-3,"m"),(1e-6,"μ"),(1e-9,"n"),(1e-12,"p")]:
        if abs(val) >= factor:
            return f"{val/factor:.3f} {prefix}{unit}"
    return f"{val:.3e} {unit}"

# ── Extract Sallen-Key sections from LP prototype (all-pole designs) ──────────
def get_sections(approx, order, fc, ripple_pass):
    wc = 2 * np.pi * fc
    if approx == "Butterworth":
        _, p, _ = signal.buttap(order)
    elif approx == "Chebyshev I":
        _, p, _ = signal.cheb1ap(order, ripple_pass)
    else:  # Bessel
        _, p, _ = signal.besselap(order, norm='phase')
    p_d  = p * wc
    done = [False] * len(p_d)
    secs = []
    for i, pi in enumerate(p_d):
        if done[i]:
            continue
        if abs(pi.imag) < abs(pi) * 1e-6:
            done[i] = True
            secs.append({"order": 1, "omega0": abs(pi.real)})
        else:
            for j in range(i + 1, len(p_d)):
                pj = p_d[j]
                if (not done[j]
                        and abs(pj.real - pi.real)           < abs(pi) * 1e-5
                        and abs(abs(pj.imag) - abs(pi.imag)) < abs(pi) * 1e-5):
                    done[i] = done[j] = True
                    omega0 = abs(pi)
                    Q = omega0 / (-2.0 * pi.real)
                    secs.append({"order": 2, "omega0": omega0, "Q": Q})
                    break
    return secs

# ── Extract biquad sections from any analog filter's poles ────────────────────
def get_biquad_sections(b, a):
    _, p, _ = signal.tf2zpk(b, a)
    done = [False] * len(p)
    secs = []
    for i, pi in enumerate(p):
        if done[i]:
            continue
        if abs(pi.imag) < abs(pi) * 1e-6:
            done[i] = True
            secs.append({"order": 1, "omega0": abs(pi.real)})
        else:
            for j in range(i + 1, len(p)):
                pj = p[j]
                if (not done[j]
                        and abs(pj.real - pi.real)           < abs(pi) * 1e-5
                        and abs(abs(pj.imag) - abs(pi.imag)) < abs(pi) * 1e-5):
                    done[i] = done[j] = True
                    omega0 = abs(pi)
                    Q = omega0 / (-2.0 * pi.real)
                    secs.append({"order": 2, "omega0": omega0, "Q": Q})
                    break
    return secs

# ── Build transfer function ───────────────────────────────────────────────────
btype = {"Low-pass": "low", "High-pass": "high",
         "Band-pass": "bandpass", "Band-stop": "bandstop"}[filter_type]

try:
    wn = [2*np.pi*f_low, 2*np.pi*f_high] if is_band else 2*np.pi*fc

    if   approx == "Butterworth":  b, a = signal.butter(order, wn, btype=btype, analog=True)
    elif approx == "Chebyshev I":  b, a = signal.cheby1(order, ripple_pass, wn, btype=btype, analog=True)
    elif approx == "Chebyshev II": b, a = signal.cheby2(order, ripple_stop, wn, btype=btype, analog=True)
    elif approx == "Elliptic":     b, a = signal.ellip(order, ripple_pass, ripple_stop, wn, btype=btype, analog=True)
    else:                          b, a = signal.bessel(order, wn, btype=btype, analog=True, norm='phase')

    # ── Frequency sweep ───────────────────────────────────────────────────────
    f_ref  = f_center if is_band else fc
    f_plot = np.logspace(np.log10(max(f_ref / 100, 0.1)), np.log10(f_ref * 100), 3000)
    _, H   = signal.freqs(b, a, 2 * np.pi * f_plot)
    mag_db = 20 * np.log10(np.abs(H) + 1e-15)
    ph_deg = np.degrees(np.unwrap(np.angle(H)))

    ph_rad = np.unwrap(np.angle(H))
    gd_ms  = -np.gradient(ph_rad, 2 * np.pi * f_plot) * 1e3
    gd_med = float(np.median(np.abs(gd_ms))) or 1.0
    gd_ms  = np.clip(gd_ms, -20 * gd_med, 20 * gd_med)

    # ── Bode plot ─────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Magnitude (dB)", "Phase (°)", "Group Delay (ms)"),
        vertical_spacing=0.08, row_heights=[0.45, 0.30, 0.25],
    )
    fig.add_trace(go.Scatter(x=f_plot, y=mag_db, line=dict(color="#2196F3", width=2.5), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_plot, y=ph_deg, line=dict(color="#FF9800", width=2.5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=f_plot, y=gd_ms,  line=dict(color="#4CAF50", width=2.5), showlegend=False), row=3, col=1)

    if is_band:
        fig.add_vline(x=f_low,    line_dash="dash", line_color="red",     annotation_text=f"f_L={f_low:.0f}Hz")
        fig.add_vline(x=f_high,   line_dash="dash", line_color="red",     annotation_text=f"f_H={f_high:.0f}Hz")
        fig.add_vline(x=f_center, line_dash="dot",  line_color="#9C27B0", annotation_text=f"f₀={f_center:.0f}Hz")
    else:
        fig.add_vline(x=fc, line_dash="dash", line_color="red", annotation_text=f"fc={fc:.0f}Hz")

    fig.add_hline(y=-3, line_dash="dot", line_color="gray", annotation_text="−3 dB")
    fig.update_xaxes(type="log", title_text="Frequency (Hz)", row=3, col=1)
    fig.update_yaxes(title_text="dB",  row=1, col=1)
    fig.update_yaxes(title_text="deg", row=2, col=1)
    fig.update_yaxes(title_text="ms",  row=3, col=1)
    freq_lbl = f"f_low={f_low:.0f}–{f_high:.0f} Hz" if is_band else f"fc={fc:.0f} Hz"
    fig.update_layout(height=680,
                      title=f"{order}-Order {approx} {filter_type}  |  {freq_lbl}",
                      margin=dict(t=60))
    st.plotly_chart(fig, use_container_width=True)

    # ── Quick metrics ─────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Approximation", approx)
    m2.metric("Order", str(order))
    m3.metric("Peak Gain", f"{float(np.max(mag_db)):.2f} dB")
    gd_pos = gd_ms[gd_ms > 0]
    m4.metric("Min Group Delay", f"{float(np.min(gd_pos)):.3f} ms" if len(gd_pos) else "—")

    st.divider()

    # ── Component values ──────────────────────────────────────────────────────
    bom_rows   = []
    all_pole   = approx in ["Butterworth", "Chebyshev I", "Bessel"]

    if not is_band and all_pole:
        # ── Sallen-Key (LP / HP) ──────────────────────────────────────────────
        st.subheader("Sallen-Key Component Values")
        st.caption("Equal-component design (R₁=R₂=R, C₁=C₂=C). Gain K sets the Q of each 2nd-order section.")
        topology = "Low-Pass" if filter_type == "Low-pass" else "High-Pass"
        sections = get_sections(approx, order, fc, ripple_pass)

        for i, sec in enumerate(sections):
            lbl = (f"Section {i+1} — {'1st' if sec['order']==1 else '2nd'} Order "
                   f"({topology} Sallen-Key)")
            with st.expander(lbl, expanded=True):
                omega0 = sec["omega0"]
                R = impedance
                if sec["order"] == 1:
                    C = 1.0 / (omega0 * R)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R",  fmt(R, "Ω"))
                    c2.metric("C",  fmt(C, "F"))
                    c3.metric("f₀", fmt(omega0 / (2*np.pi), "Hz"))
                    bom_rows += [{"Section": i+1, "Component": "R",  "Value": R, "Unit": "Ω"},
                                 {"Section": i+1, "Component": "C",  "Value": C, "Unit": "F"}]
                else:
                    Q  = sec["Q"]
                    C  = 1.0 / (omega0 * R)
                    K  = 3.0 - 1.0 / Q
                    Rg = R
                    Rf = (K - 1.0) * Rg
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("R₁ = R₂", fmt(R,  "Ω"))
                    c2.metric("C₁ = C₂", fmt(C,  "F"))
                    c3.metric("f₀",       fmt(omega0 / (2*np.pi), "Hz"))
                    c4.metric("Q",        f"{Q:.4f}")
                    c5.metric("Gain K",   f"{K:.4f}")
                    if K >= 1.0:
                        st.caption(
                            f"Gain network: Rg = {fmt(Rg,'Ω')},  Rf = {fmt(Rf,'Ω')}  "
                            f"→  K = 1+Rf/Rg = {K:.4f}"
                        )
                        bom_rows += [
                            {"Section": i+1, "Component": "R1", "Value": R,  "Unit": "Ω"},
                            {"Section": i+1, "Component": "R2", "Value": R,  "Unit": "Ω"},
                            {"Section": i+1, "Component": "C1", "Value": C,  "Unit": "F"},
                            {"Section": i+1, "Component": "C2", "Value": C,  "Unit": "F"},
                            {"Section": i+1, "Component": "Rg", "Value": Rg, "Unit": "Ω"},
                            {"Section": i+1, "Component": "Rf", "Value": Rf, "Unit": "Ω"},
                        ]
                    else:
                        st.warning(f"K = {K:.4f} < 1 — use unity-gain (voltage-follower) configuration.")

    elif not is_band and not all_pole:
        st.info(
            f"ℹ️  **{approx}** has finite transmission zeros. "
            "Sallen-Key (all-pole) doesn't apply. "
            "Use Fleischer-Tow or friend sections instead."
        )

    elif filter_type == "Band-pass":
        # ── Tow-Thomas (State-Variable) biquad ───────────────────────────────
        st.subheader("Tow-Thomas (State-Variable) Biquad Component Values")
        st.caption(
            "3 op-amp topology. LP, BP, and HP outputs are all available simultaneously. "
            "Equal integrator resistors R and capacitors C; R_Q sets the Q factor."
        )
        bq_secs = get_biquad_sections(b, a)
        n2 = sum(1 for s in bq_secs if s["order"] == 2)
        n1 = sum(1 for s in bq_secs if s["order"] == 1)
        st.caption(
            f"Decomposed into **{n2} biquad section(s)**"
            + (f" + {n1} first-order section(s)" if n1 else "") + "."
        )

        for i, sec in enumerate(bq_secs):
            if sec["order"] == 2:
                omega0 = sec["omega0"]
                Q      = sec["Q"]
                R      = impedance
                C      = 1.0 / (omega0 * R)
                R_Q    = Q * R      # Q-setting resistor
                R_in   = R_Q        # input resistor → unity passband gain

                lbl = f"Section {i+1} — 2nd Order Band-Pass (Tow-Thomas)"
                with st.expander(lbl, expanded=True):
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("R_int (×2)", fmt(R,   "Ω"))
                    c2.metric("C (×2)",     fmt(C,   "F"))
                    c3.metric("R_Q = R_in", fmt(R_Q, "Ω"))
                    c4.metric("f₀",         fmt(omega0 / (2*np.pi), "Hz"))
                    c5.metric("Q",          f"{Q:.4f}")
                    st.caption(
                        f"Passband gain = R_Q / R_in = 1 (0 dB)  ·  "
                        f"τ_int = R·C = {R*C*1e6:.2f} μs"
                    )
                    bom_rows += [
                        {"Section": i+1, "Component": "R_int1", "Value": R,    "Unit": "Ω"},
                        {"Section": i+1, "Component": "R_int2", "Value": R,    "Unit": "Ω"},
                        {"Section": i+1, "Component": "C1",     "Value": C,    "Unit": "F"},
                        {"Section": i+1, "Component": "C2",     "Value": C,    "Unit": "F"},
                        {"Section": i+1, "Component": "R_Q",    "Value": R_Q,  "Unit": "Ω"},
                        {"Section": i+1, "Component": "R_in",   "Value": R_in, "Unit": "Ω"},
                    ]
            else:
                omega0 = sec["omega0"]
                R = impedance
                C = 1.0 / (omega0 * R)
                with st.expander(f"Section {i+1} — 1st Order", expanded=True):
                    c1, c2 = st.columns(2)
                    c1.metric("R", fmt(R, "Ω"))
                    c2.metric("C", fmt(C, "F"))

    else:
        # ── Band-stop: show section parameters ───────────────────────────────
        st.subheader("Band-Stop Section Parameters")
        bq_secs  = get_biquad_sections(b, a)
        rows_bs  = []
        for i, sec in enumerate(bq_secs):
            if sec["order"] == 2:
                omega0 = sec["omega0"]
                Q_sec  = sec["Q"]
                C      = 1.0 / (omega0 * impedance)
                rows_bs.append({
                    "Section":     i + 1,
                    "f₀ (Hz)":     f"{omega0/(2*np.pi):.2f}",
                    "Q":           f"{Q_sec:.4f}",
                    "R (base)":    fmt(impedance, "Ω"),
                    "C = 1/(ω₀R)": fmt(C, "F"),
                })
        if rows_bs:
            st.dataframe(rows_bs, use_container_width=True, hide_index=True)
        topo_note = (
            "ℹ️  High-Q notch (Q ≥ 5): **Twin-T** or **Bainter** topology recommended — Phase 4."
            if Q_band >= 5 else
            "ℹ️  Wideband notch: realize as parallel LP + HP paths — Phase 4."
        )
        st.info(topo_note)

    # ── Bill of Materials ─────────────────────────────────────────────────────
    if bom_rows:
        st.subheader("Bill of Materials")
        df = pd.DataFrame(bom_rows)
        df["Formatted Value"] = df.apply(lambda r: fmt(r["Value"], r["Unit"]), axis=1)
        df["Value (SI)"]      = df["Value"].apply(lambda v: f"{v:.6e}")

        st.dataframe(
            df[["Section", "Component", "Formatted Value"]],
            use_container_width=True, hide_index=True,
        )
        csv_str = df[["Section", "Component", "Value (SI)", "Unit"]].to_csv(index=False)
        st.download_button(
            label="⬇ Download BOM (CSV)",
            data=csv_str,
            file_name=f"bom_{approx}_{filter_type}_order{order}.csv",
            mime="text/csv",
        )

    # ── Monte Carlo Tolerance Analysis ────────────────────────────────────────
    if not is_band and all_pole:
        with st.expander("Monte Carlo Tolerance Analysis", expanded=False):
            tc, nc = st.columns(2)
            tol_pct = tc.slider("Component Tolerance (%)", 1, 20, 5, key="mc_tol")
            n_runs  = nc.select_slider("Simulations", [100, 200, 500, 1000], value=200, key="mc_n")
            tol = tol_pct / 100

            sections_mc = get_sections(approx, order, fc, ripple_pass)
            f_mc = np.logspace(np.log10(max(fc / 100, 0.1)), np.log10(fc * 100), 800)
            rng  = np.random.default_rng(42)
            mc_mag = np.zeros((n_runs, len(f_mc)))

            for run in range(n_runs):
                b_t = np.array([1.0])
                a_t = np.array([1.0])
                for sec in sections_mc:
                    o = sec["omega0"] * (1 + rng.uniform(-tol, tol))
                    if sec["order"] == 1:
                        b_s = [o]      if filter_type == "Low-pass" else [1, 0]
                        a_s = [1, o]
                    else:
                        q   = sec["Q"] * (1 + rng.uniform(-tol, tol))
                        b_s = [o**2]   if filter_type == "Low-pass" else [1, 0, 0]
                        a_s = [1, o/q, o**2]
                    b_t = np.polymul(b_t, b_s)
                    a_t = np.polymul(a_t, a_s)
                _, H_r = signal.freqs(b_t, a_t, 2*np.pi*f_mc)
                mc_mag[run] = 20 * np.log10(np.abs(H_r) + 1e-15)

            mc_mean = mc_mag.mean(axis=0)
            mc_p5   = np.percentile(mc_mag,  5, axis=0)
            mc_p95  = np.percentile(mc_mag, 95, axis=0)
            mc_min  = mc_mag.min(axis=0)
            mc_max  = mc_mag.max(axis=0)

            _, H_nom = signal.freqs(b, a, 2*np.pi*f_mc)
            nom_db   = 20 * np.log10(np.abs(H_nom) + 1e-15)

            mc_fig = go.Figure()
            mc_fig.add_trace(go.Scatter(
                x=np.concatenate([f_mc, f_mc[::-1]]),
                y=np.concatenate([mc_max, mc_min[::-1]]),
                fill="toself", fillcolor="rgba(33,150,243,0.10)",
                line=dict(width=0), name="Min–Max",
            ))
            mc_fig.add_trace(go.Scatter(
                x=np.concatenate([f_mc, f_mc[::-1]]),
                y=np.concatenate([mc_p95, mc_p5[::-1]]),
                fill="toself", fillcolor="rgba(33,150,243,0.25)",
                line=dict(width=0), name="5th–95th pct",
            ))
            mc_fig.add_trace(go.Scatter(x=f_mc, y=mc_mean,
                line=dict(color="#2196F3", width=2), name="Mean"))
            mc_fig.add_trace(go.Scatter(x=f_mc, y=nom_db,
                line=dict(color="#F44336", width=2, dash="dash"), name="Nominal"))
            mc_fig.add_vline(x=fc, line_dash="dash", line_color="gray")
            mc_fig.add_hline(y=-3, line_dash="dot",  line_color="gray", annotation_text="−3 dB")
            mc_fig.update_xaxes(type="log", title_text="Frequency (Hz)")
            mc_fig.update_yaxes(title_text="Magnitude (dB)")
            mc_fig.update_layout(
                height=420,
                title=f"Monte Carlo — {n_runs} runs · ±{tol_pct}% R/C tolerance",
                legend=dict(x=0.02, y=0.02),
            )
            st.plotly_chart(mc_fig, use_container_width=True)

            # −3 dB frequency statistics
            f3db_list = []
            for resp in mc_mag:
                if filter_type == "Low-pass":
                    idx = np.where(resp <= -3)[0]
                    if len(idx): f3db_list.append(f_mc[idx[0]])
                else:
                    idx = np.where(resp >= -3)[0]
                    if len(idx): f3db_list.append(f_mc[idx[0]])
            if f3db_list:
                s1, s2, s3 = st.columns(3)
                s1.metric("Mean f₋₃dB", fmt(float(np.mean(f3db_list)), "Hz"))
                s2.metric("Std Dev",     fmt(float(np.std(f3db_list)),  "Hz"))
                s3.metric("Spread", f"{fmt(min(f3db_list), 'Hz')} – {fmt(max(f3db_list), 'Hz')}")

    # ── Step & Impulse Response ───────────────────────────────────────────────
    with st.expander("Step & Impulse Response", expanded=False):
        t_end = 10.0 / bw if is_band else 5.0 / fc
        t_vec = np.linspace(0, t_end, 2000)
        try:
            _, y_step = signal.step(   (b, a), T=t_vec)
            _, y_imp  = signal.impulse((b, a), T=t_vec)
            t_ms = t_vec * 1e3
            rf = make_subplots(rows=1, cols=2, subplot_titles=("Step Response", "Impulse Response"))
            rf.add_trace(go.Scatter(x=t_ms, y=y_step,
                line=dict(color="#2196F3", width=2), showlegend=False), row=1, col=1)
            rf.add_trace(go.Scatter(x=t_ms, y=y_imp,
                line=dict(color="#FF9800", width=2), showlegend=False), row=1, col=2)
            rf.update_xaxes(title_text="Time (ms)")
            rf.update_layout(height=350, showlegend=False)
            st.plotly_chart(rf, use_container_width=True)
        except Exception as te:
            st.warning(f"Time-domain response unavailable: {te}")

    # ── Pole-Zero Map ─────────────────────────────────────────────────────────
    with st.expander("Pole-Zero Map", expanded=False):
        z_arr, p_arr, _ = signal.tf2zpk(b, a)
        wc_ref = 2 * np.pi * (f_center if is_band else fc)
        pz = go.Figure()
        pz.add_shape(type="circle", x0=-1, y0=-1, x1=1, y1=1,
                     line=dict(color="lightgray", dash="dot"))
        if len(z_arr):
            pz.add_trace(go.Scatter(
                x=z_arr.real / wc_ref, y=z_arr.imag / wc_ref,
                mode="markers", marker=dict(symbol="circle-open", size=12, color="#2196F3"),
                name="Zeros"))
        pz.add_trace(go.Scatter(
            x=p_arr.real / wc_ref, y=p_arr.imag / wc_ref,
            mode="markers", marker=dict(symbol="x", size=12, color="#F44336"),
            name="Poles"))
        pz.update_layout(height=420, xaxis_title="Re(s) / ωc", yaxis_title="Im(s) / ωc",
                         yaxis_scaleanchor="x", legend=dict(x=0.02, y=0.98))
        st.plotly_chart(pz, use_container_width=True)

except Exception as e:
    st.error(f"Computation error: {e}")

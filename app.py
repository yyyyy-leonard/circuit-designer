import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Active Filter Designer", layout="wide")
st.title("Active Filter Designer")
st.caption("Design an analogue filter, get component values, and check how much "
           "component tolerance will move the response before you build it.")

# sidebar inputs
with st.sidebar:
    st.header("Filter Parameters")
    filter_type = st.selectbox("Type", ["Low-pass", "High-pass", "Band-pass", "Band-stop"])
    approx = st.selectbox("Approximation",
                          ["Butterworth", "Chebyshev I", "Chebyshev II", "Elliptic", "Bessel"])
    order = st.slider("Order", 1, 8, 2)

    is_band = filter_type in ["Band-pass", "Band-stop"]

    if is_band:
        st.markdown("**Frequency Band**")
        ca, cb = st.columns(2)
        f_low = ca.number_input("f_low (Hz)", 1.0, 1e5, 800.0, step=10.0)
        f_high = cb.number_input("f_high (Hz)", 1.0, 1e5, 1200.0, step=10.0)
        if f_high <= f_low:
            st.error("f_high must be greater than f_low")
            st.stop()
        f_center = float(np.sqrt(f_low * f_high))  # geometric centre
        bw = f_high - f_low
        Q_band = f_center / bw
        st.caption(f"f₀ = {f_center:.1f} Hz · BW = {bw:.1f} Hz · Q = {Q_band:.2f}")
        fc = f_center
    else:
        fc = st.number_input("Cutoff Frequency (Hz)", 1.0, 1e5, 1000.0, step=100.0)

    impedance = st.number_input("Base Impedance R (Ω)", 100.0, 1e6, 10000.0, step=1000.0)
    ripple_pass = 0.5
    ripple_stop = 40.0
    if approx in ["Chebyshev I", "Elliptic"]:
        ripple_pass = st.slider("Passband Ripple (dB)", 0.1, 3.0, 0.5, step=0.1)
    if approx in ["Chebyshev II", "Elliptic"]:
        ripple_stop = st.slider("Stopband Attenuation (dB)", 20.0, 80.0, 40.0, step=1.0)


def fmt(val, unit):
    """Format a value with an SI prefix, e.g. 1.5e-8 F -> '15.000 nF'."""
    for factor, prefix in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"),
                           (1, ""), (1e-3, "m"), (1e-6, "μ"), (1e-9, "n"), (1e-12, "p")]:
        if abs(val) >= factor:
            return f"{val/factor:.3f} {prefix}{unit}"
    return f"{val:.3e} {unit}"


def pair_poles(poles):
    """Group poles into 1st-order (real) and 2nd-order (conjugate pair) sections."""
    done = [False] * len(poles)
    secs = []
    for i, pi in enumerate(poles):
        if done[i]:
            continue
        if abs(pi.imag) < abs(pi) * 1e-6:
            done[i] = True
            secs.append({"order": 1, "omega0": abs(pi.real)})
            continue
        for j in range(i + 1, len(poles)):
            pj = poles[j]
            if (not done[j]
                    and abs(pj.real - pi.real) < abs(pi) * 1e-5
                    and abs(abs(pj.imag) - abs(pi.imag)) < abs(pi) * 1e-5):
                done[i] = done[j] = True
                omega0 = abs(pi)
                secs.append({"order": 2, "omega0": omega0, "Q": omega0 / (-2.0 * pi.real)})
                break
    return secs


def get_sections(approx, order, fc, ripple_pass):
    """Sections for an all-pole LP prototype scaled to fc (used for Sallen-Key)."""
    if approx == "Butterworth":
        _, p, _ = signal.buttap(order)
    elif approx == "Chebyshev I":
        _, p, _ = signal.cheb1ap(order, ripple_pass)
    else:  # Bessel
        _, p, _ = signal.besselap(order, norm='phase')
    return pair_poles(p * 2 * np.pi * fc)


def get_biquad_sections(b, a):
    _, p, _ = signal.tf2zpk(b, a)
    return pair_poles(p)


def sallen_key_w0_q(R1, R2, C1, C2, K, highpass=False):
    """w0 and Q of a Sallen-Key stage from actual component values.

    Needed for the Monte Carlo: with unequal (toleranced) parts the
    simple equal-component formula K = 3 - 1/Q no longer holds.
    """
    w0 = 1.0 / np.sqrt(R1 * R2 * C1 * C2)
    root = np.sqrt(R1 * R2 * C1 * C2)
    if highpass:
        inv_q = (R1 * C1 + R1 * C2 + R2 * C2 * (1 - K)) / root
    else:
        inv_q = (R1 * C1 * (1 - K) + C2 * (R1 + R2)) / root
    return w0, 1.0 / inv_q


btype = {"Low-pass": "low", "High-pass": "high",
         "Band-pass": "bandpass", "Band-stop": "bandstop"}[filter_type]

try:
    wn = [2*np.pi*f_low, 2*np.pi*f_high] if is_band else 2*np.pi*fc

    if approx == "Butterworth":
        b, a = signal.butter(order, wn, btype=btype, analog=True)
    elif approx == "Chebyshev I":
        b, a = signal.cheby1(order, ripple_pass, wn, btype=btype, analog=True)
    elif approx == "Chebyshev II":
        b, a = signal.cheby2(order, ripple_stop, wn, btype=btype, analog=True)
    elif approx == "Elliptic":
        b, a = signal.ellip(order, ripple_pass, ripple_stop, wn, btype=btype, analog=True)
    else:
        b, a = signal.bessel(order, wn, btype=btype, analog=True, norm='phase')

    # frequency response, 2 decades either side of fc
    f_ref = f_center if is_band else fc
    f_plot = np.logspace(np.log10(max(f_ref / 100, 0.1)), np.log10(f_ref * 100), 3000)
    _, H = signal.freqs(b, a, 2 * np.pi * f_plot)
    mag_db = 20 * np.log10(np.abs(H) + 1e-15)
    ph_rad = np.unwrap(np.angle(H))
    ph_deg = np.degrees(ph_rad)

    # group delay = -dφ/dω, clipped so spikes near the notch don't wreck the y axis
    gd_ms = -np.gradient(ph_rad, 2 * np.pi * f_plot) * 1e3
    gd_med = float(np.median(np.abs(gd_ms))) or 1.0
    gd_ms = np.clip(gd_ms, -20 * gd_med, 20 * gd_med)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Magnitude (dB)", "Phase (°)", "Group Delay (ms)"),
        vertical_spacing=0.08, row_heights=[0.45, 0.30, 0.25],
    )
    fig.add_trace(go.Scatter(x=f_plot, y=mag_db, line=dict(color="#2196F3", width=2.5), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_plot, y=ph_deg, line=dict(color="#FF9800", width=2.5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=f_plot, y=gd_ms, line=dict(color="#4CAF50", width=2.5), showlegend=False), row=3, col=1)

    if is_band:
        fig.add_vline(x=f_low, line_dash="dash", line_color="red", annotation_text=f"f_L={f_low:.0f}Hz")
        fig.add_vline(x=f_high, line_dash="dash", line_color="red", annotation_text=f"f_H={f_high:.0f}Hz")
        fig.add_vline(x=f_center, line_dash="dot", line_color="#9C27B0", annotation_text=f"f₀={f_center:.0f}Hz")
    else:
        fig.add_vline(x=fc, line_dash="dash", line_color="red", annotation_text=f"fc={fc:.0f}Hz")

    fig.add_hline(y=-3, line_dash="dot", line_color="gray", annotation_text="−3 dB", row=1, col=1)
    fig.update_xaxes(type="log")  # all three rows, otherwise the fc markers land in the wrong place
    fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=1)
    fig.update_yaxes(title_text="dB", row=1, col=1)
    fig.update_yaxes(title_text="deg", row=2, col=1)
    fig.update_yaxes(title_text="ms", row=3, col=1)
    freq_lbl = f"{f_low:.0f}–{f_high:.0f} Hz" if is_band else f"fc={fc:.0f} Hz"
    fig.update_layout(height=680,
                      title=f"Order {order} {approx} {filter_type}  |  {freq_lbl}",
                      margin=dict(t=60))
    st.plotly_chart(fig, width="stretch")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Approximation", approx)
    m2.metric("Order", str(order))
    m3.metric("Peak Gain", f"{float(np.max(mag_db)):.2f} dB")
    gd_pos = gd_ms[gd_ms > 0]
    m4.metric("Min Group Delay", f"{float(np.min(gd_pos)):.3f} ms" if len(gd_pos) else "—")

    st.divider()

    # component values
    bom_rows = []
    all_pole = approx in ["Butterworth", "Chebyshev I", "Bessel"]

    if not is_band and all_pole:
        st.subheader("Sallen-Key Component Values")
        st.caption("Equal-component design (R₁=R₂=R, C₁=C₂=C). Gain K sets the Q of each 2nd-order section.")
        topology = "Low-Pass" if filter_type == "Low-pass" else "High-Pass"
        sections = get_sections(approx, order, fc, ripple_pass)

        for i, sec in enumerate(sections):
            lbl = (f"Section {i+1}: {'1st' if sec['order'] == 1 else '2nd'} order "
                   f"({topology} Sallen-Key)")
            with st.expander(lbl, expanded=True):
                omega0 = sec["omega0"]
                R = impedance
                C = 1.0 / (omega0 * R)
                if sec["order"] == 1:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R", fmt(R, "Ω"))
                    c2.metric("C", fmt(C, "F"))
                    c3.metric("f₀", fmt(omega0 / (2*np.pi), "Hz"))
                    bom_rows += [{"Section": i+1, "Component": "R", "Value": R, "Unit": "Ω"},
                                 {"Section": i+1, "Component": "C", "Value": C, "Unit": "F"}]
                else:
                    Q = sec["Q"]
                    K = 3.0 - 1.0 / Q   # equal-component Sallen-Key: Q = 1/(3-K)
                    Rg = R
                    Rf = (K - 1.0) * Rg
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("R₁ = R₂", fmt(R, "Ω"))
                    c2.metric("C₁ = C₂", fmt(C, "F"))
                    c3.metric("f₀", fmt(omega0 / (2*np.pi), "Hz"))
                    c4.metric("Q", f"{Q:.4f}")
                    c5.metric("Gain K", f"{K:.4f}")
                    if K >= 1.0:
                        st.caption(
                            f"Gain network: Rg = {fmt(Rg, 'Ω')},  Rf = {fmt(Rf, 'Ω')}  "
                            f"→  K = 1+Rf/Rg = {K:.4f}"
                        )
                        bom_rows += [
                            {"Section": i+1, "Component": "R1", "Value": R, "Unit": "Ω"},
                            {"Section": i+1, "Component": "R2", "Value": R, "Unit": "Ω"},
                            {"Section": i+1, "Component": "C1", "Value": C, "Unit": "F"},
                            {"Section": i+1, "Component": "C2", "Value": C, "Unit": "F"},
                            {"Section": i+1, "Component": "Rg", "Value": Rg, "Unit": "Ω"},
                            {"Section": i+1, "Component": "Rf", "Value": Rf, "Unit": "Ω"},
                        ]
                    else:
                        st.warning(f"K = {K:.4f} < 1, so use a unity-gain (voltage follower) stage instead.")

    elif not is_band and not all_pole:
        st.info(
            f"**{approx}** has finite transmission zeros, so the all-pole Sallen-Key "
            "stage can't realise it. A Fleischer-Tow or Friend biquad would be needed "
            "(not implemented yet)."
        )

    elif filter_type == "Band-pass":
        # Tow-Thomas biquad: 3 op-amps, gives LP/BP/HP outputs at the same time
        st.subheader("Tow-Thomas Biquad Component Values")
        st.caption(
            "3 op-amp topology. Equal integrator resistors R and capacitors C; R_Q sets the Q."
        )
        bq_secs = get_biquad_sections(b, a)
        n2 = sum(1 for s in bq_secs if s["order"] == 2)
        n1 = sum(1 for s in bq_secs if s["order"] == 1)
        st.caption(
            f"Split into **{n2} biquad section(s)**"
            + (f" + {n1} first-order section(s)" if n1 else "") + "."
        )

        for i, sec in enumerate(bq_secs):
            omega0 = sec["omega0"]
            R = impedance
            C = 1.0 / (omega0 * R)
            if sec["order"] == 2:
                Q = sec["Q"]
                R_Q = Q * R
                R_in = R_Q  # gives 0 dB passband gain

                with st.expander(f"Section {i+1}: 2nd order band-pass (Tow-Thomas)", expanded=True):
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("R_int (×2)", fmt(R, "Ω"))
                    c2.metric("C (×2)", fmt(C, "F"))
                    c3.metric("R_Q = R_in", fmt(R_Q, "Ω"))
                    c4.metric("f₀", fmt(omega0 / (2*np.pi), "Hz"))
                    c5.metric("Q", f"{Q:.4f}")
                    st.caption(
                        f"Passband gain = R_Q / R_in = 1 (0 dB)  ·  "
                        f"τ_int = R·C = {R*C*1e6:.2f} μs"
                    )
                    bom_rows += [
                        {"Section": i+1, "Component": "R_int1", "Value": R, "Unit": "Ω"},
                        {"Section": i+1, "Component": "R_int2", "Value": R, "Unit": "Ω"},
                        {"Section": i+1, "Component": "C1", "Value": C, "Unit": "F"},
                        {"Section": i+1, "Component": "C2", "Value": C, "Unit": "F"},
                        {"Section": i+1, "Component": "R_Q", "Value": R_Q, "Unit": "Ω"},
                        {"Section": i+1, "Component": "R_in", "Value": R_in, "Unit": "Ω"},
                    ]
            else:
                with st.expander(f"Section {i+1}: 1st order", expanded=True):
                    c1, c2 = st.columns(2)
                    c1.metric("R", fmt(R, "Ω"))
                    c2.metric("C", fmt(C, "F"))

    else:
        # band-stop: only section parameters for now, no topology yet
        st.subheader("Band-Stop Section Parameters")
        rows_bs = []
        for i, sec in enumerate(get_biquad_sections(b, a)):
            if sec["order"] == 2:
                omega0 = sec["omega0"]
                rows_bs.append({
                    "Section": i + 1,
                    "f₀ (Hz)": f"{omega0/(2*np.pi):.2f}",
                    "Q": f"{sec['Q']:.4f}",
                    "R (base)": fmt(impedance, "Ω"),
                    "C = 1/(ω₀R)": fmt(1.0 / (omega0 * impedance), "F"),
                })
        if rows_bs:
            st.dataframe(rows_bs, width="stretch", hide_index=True)
        if Q_band >= 5:
            st.info("High-Q notch (Q ≥ 5): a Twin-T or Bainter notch would suit this. Not implemented yet.")
        else:
            st.info("Wide notch: could be built as parallel LP + HP paths. Not implemented yet.")

    if bom_rows:
        st.subheader("Bill of Materials")
        df = pd.DataFrame(bom_rows)
        df["Formatted Value"] = df.apply(lambda r: fmt(r["Value"], r["Unit"]), axis=1)
        df["Value (SI)"] = df["Value"].apply(lambda v: f"{v:.6e}")

        st.dataframe(df[["Section", "Component", "Formatted Value"]],
                     width="stretch", hide_index=True)
        csv_str = df[["Section", "Component", "Value (SI)", "Unit"]].to_csv(index=False)
        st.download_button(
            label="Download BOM (CSV)",
            data=csv_str,
            file_name=f"bom_{approx}_{filter_type}_order{order}.csv",
            mime="text/csv",
        )

    # Monte Carlo: every R and C gets its own random error, then w0/Q are
    # recomputed from the actual values (so tolerance in K also shows up)
    if not is_band and all_pole:
        with st.expander("Monte Carlo Tolerance Analysis", expanded=False):
            tc, nc = st.columns(2)
            tol_pct = tc.slider("Component Tolerance (%)", 1, 20, 5, key="mc_tol")
            n_runs = nc.select_slider("Simulations", [100, 200, 500, 1000, 5000, 10000],
                                      value=1000, key="mc_n")
            tol = tol_pct / 100
            hp = filter_type == "High-pass"

            sections_mc = get_sections(approx, order, fc, ripple_pass)
            f_mc = np.logspace(np.log10(max(fc / 10, 0.1)), np.log10(fc * 10), 800)  # zoom in on the corner
            rng = np.random.default_rng(42)  # fixed seed so results don't jump on every rerun
            mc_mag = np.zeros((n_runs, len(f_mc)))
            n_unstable = 0

            def vary(x):
                return x * (1 + rng.uniform(-tol, tol))

            for run in range(n_runs):
                b_t = np.array([1.0])
                a_t = np.array([1.0])
                stable = True
                for sec in sections_mc:
                    R = impedance
                    C = 1.0 / (sec["omega0"] * R)
                    if sec["order"] == 1:
                        o = 1.0 / (vary(R) * vary(C))
                        b_s = [1, 0] if hp else [o]
                        a_s = [1, o]
                    else:
                        K_nom = 3.0 - 1.0 / sec["Q"]
                        Rg = vary(R)
                        Rf = vary((K_nom - 1.0) * R)
                        K = 1 + Rf / Rg
                        o, q = sallen_key_w0_q(vary(R), vary(R), vary(C), vary(C), K, highpass=hp)
                        if q <= 0:  # K pushed past 3 -> poles in the right half plane
                            stable = False
                        # normalise to passband gain 1 so curves line up with nominal
                        b_s = [1, 0, 0] if hp else [o**2]
                        a_s = [1, o/q, o**2]
                    b_t = np.polymul(b_t, b_s)
                    a_t = np.polymul(a_t, a_s)
                if not stable:
                    n_unstable += 1
                    mc_mag[run] = np.nan
                    continue
                _, H_r = signal.freqs(b_t, a_t, 2*np.pi*f_mc)
                mc_mag[run] = 20 * np.log10(np.abs(H_r) + 1e-15)

            if n_unstable:
                st.warning(f"{n_unstable} of {n_runs} runs were unstable (left out of the plot). "
                           "Equal-component Sallen-Key needs K close to 3 at high Q, so small "
                           "resistor errors can make it oscillate.")
            mc_mag = mc_mag[~np.isnan(mc_mag[:, 0])]
            if len(mc_mag) == 0:
                st.stop()

            mc_mean = mc_mag.mean(axis=0)
            mc_p5 = np.percentile(mc_mag, 5, axis=0)
            mc_p95 = np.percentile(mc_mag, 95, axis=0)
            mc_min = mc_mag.min(axis=0)
            mc_max = mc_mag.max(axis=0)

            _, H_nom = signal.freqs(b, a, 2*np.pi*f_mc)
            nom_db = 20 * np.log10(np.abs(H_nom) + 1e-15)

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
            mc_fig.add_hline(y=-3, line_dash="dot", line_color="gray", annotation_text="−3 dB")
            mc_fig.update_xaxes(type="log", title_text="Frequency (Hz)")
            mc_fig.update_yaxes(title_text="Magnitude (dB)", range=[-40, max(3.0, float(mc_max.max()) + 1)])
            mc_fig.update_layout(
                height=420,
                title=f"Monte Carlo: {n_runs} runs, ±{tol_pct}% on every R and C",
                legend=dict(x=0.02, y=0.02),
            )
            st.plotly_chart(mc_fig, width="stretch")

            # where each run crosses -3 dB
            f3db_list = []
            for resp in mc_mag:
                idx = np.where(resp >= -3)[0] if hp else np.where(resp <= -3)[0]
                if len(idx):
                    f3db_list.append(f_mc[idx[0]])
            if f3db_list:
                f3 = np.array(f3db_list)
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Mean f₋₃dB", fmt(float(f3.mean()), "Hz"))
                s2.metric("Std Dev", fmt(float(f3.std()), "Hz"))
                s3.metric("Range", f"{f3.min():.0f} – {f3.max():.0f} Hz")
                s4.metric("Peak gain (worst run)", f"{mc_mag.max():.2f} dB")

    with st.expander("Step & Impulse Response", expanded=False):
        t_end = 10.0 / bw if is_band else 5.0 / fc
        t_vec = np.linspace(0, t_end, 2000)
        try:
            _, y_step = signal.step((b, a), T=t_vec)
            _, y_imp = signal.impulse((b, a), T=t_vec)
            t_ms = t_vec * 1e3
            rf = make_subplots(rows=1, cols=2, subplot_titles=("Step Response", "Impulse Response"))
            rf.add_trace(go.Scatter(x=t_ms, y=y_step,
                                    line=dict(color="#2196F3", width=2), showlegend=False), row=1, col=1)
            rf.add_trace(go.Scatter(x=t_ms, y=y_imp,
                                    line=dict(color="#FF9800", width=2), showlegend=False), row=1, col=2)
            rf.update_xaxes(title_text="Time (ms)")
            rf.update_layout(height=350, showlegend=False)
            st.plotly_chart(rf, width="stretch")
        except Exception as te:
            st.warning(f"Time-domain response unavailable: {te}")

    with st.expander("Pole-Zero Map", expanded=False):
        z_arr, p_arr, _ = signal.tf2zpk(b, a)
        wc_ref = 2 * np.pi * f_ref  # normalise so the unit circle is at fc
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
        st.plotly_chart(pz, width="stretch")

except Exception as e:
    st.error(f"Computation error: {e}")

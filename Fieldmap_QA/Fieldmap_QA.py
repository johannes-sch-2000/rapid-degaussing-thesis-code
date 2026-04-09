import os
import json
import math
import csv
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
    PLOTLY_ERR = None
except Exception as e:
    go = None
    PLOTLY_OK = False
    PLOTLY_ERR = str(e)

try:
    import tkinter as tk
    from tkinter import filedialog
    TK_OK = True
except Exception:
    TK_OK = False


# ============================================================
# Utilities
# ============================================================

def _read_csv_dicts(path: str):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def load_raw_map_from_summary(session_dir: str, scale_nt_per_v: np.ndarray):
    """
    Loads points from summary_raw.csv (raw volts), converts B to nT using scale_nt_per_v.
    Returns: x,y,z,Bx,By,Bz,Bmag in nT, plus a label.
    """
    p = os.path.join(session_dir, "summary_raw.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}. (Need a raw-only export / end-session without offsets)")

    rows = _read_csv_dicts(p)
    if len(rows) == 0:
        raise RuntimeError("summary_raw.csv is empty.")

    xs, ys, zs = [], [], []
    bx_v, by_v, bz_v = [], [], []

    for r in rows:
        try:
            xs.append(float(r["x"]))
            ys.append(float(r["y"]))
            zs.append(float(r["z"]))
        except Exception:
            raise RuntimeError("summary_raw.csv missing x/y/z columns (meters).")

        if "Bx_V" in r:
            bx_v.append(float(r["Bx_V"]))
            by_v.append(float(r["By_V"]))
            bz_v.append(float(r["Bz_V"]))
        elif "Bx" in r:
            bx_v.append(float(r["Bx"]))
            by_v.append(float(r["By"]))
            bz_v.append(float(r["Bz"]))
        else:
            raise RuntimeError("summary_raw.csv missing Bx_V/By_V/Bz_V columns.")

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)

    bv = np.column_stack([bx_v, by_v, bz_v]).astype(float)
    scale = np.asarray(scale_nt_per_v, dtype=float).reshape(3,)
    b_nt = bv * scale[None, :]  

    bx = b_nt[:, 0]
    by = b_nt[:, 1]
    bz = b_nt[:, 2]
    bmag = np.sqrt(bx*bx + by*by + bz*bz)

    return xs, ys, zs, bx, by, bz, bmag, "summary_raw.csv (no offset)"

def plot_cones_raw_plotly(session_dir: str, x, y, z, bx, by, bz, bmag,
                          title: str = "Raw field map (no offset)",
                          cmin: float = 0.0,
                          cmax: float = None,
                          sizemode: str = "scaled",
                          sizeref: float = 0.6):
    """
    Plotly cone plot with per-map cone scaling and fixed color scale.
    """
    if not PLOTLY_OK:
        raise RuntimeError(f"Plotly not available: {PLOTLY_ERR}")

    if cmax is None:
        cmax = float(np.nanmax(bmag)) if np.isfinite(np.nanmax(bmag)) else 1.0
    if cmax <= cmin:
        cmax = cmin + 1.0

    pad = 0.05
    xr = [float(np.min(x) - pad), float(np.max(x) + pad)]
    yr = [float(np.min(y) - pad), float(np.max(y) + pad)]
    zr = [float(np.min(z) - pad), float(np.max(z) + pad)]

    xticks = sorted(np.unique(np.round(x, 10)).tolist())
    yticks = sorted(np.unique(np.round(y, 10)).tolist())
    zticks = sorted(np.unique(np.round(z, 10)).tolist())

    fig = go.Figure(
        data=go.Cone(
            x=x, y=y, z=z,
            u=bx, v=by, w=bz,
            colorscale="Jet",
            cmin=cmin, cmax=cmax,
            sizemode=sizemode,
            sizeref=sizeref,
            anchor="cm",
            showscale=True,
            colorbar=dict(
                title=None,
                thickness=22,
                len=0.72,
                x=0.90,
                y=0.50,
                tickfont=dict(size=16),
            )
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                family="Latin Modern Roman, Computer Modern, CMU Serif, STIX Two Text, Times New Roman, serif",
                size=32
            )
        ),
        font=dict(
            family="Latin Modern Roman, Computer Modern, CMU Serif, STIX Two Text, Times New Roman, serif",
            size=26
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text="x (m)", font=dict(size=28)),
                range=xr,
                showbackground=False,
                showgrid=True,
                gridcolor="rgba(120,120,120,0.35)",
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor="rgba(80,80,80,0.8)",
                tickvals=xticks,
                tickfont=dict(size=18),
            ),
            yaxis=dict(
                title=dict(text="y (m)", font=dict(size=28)),
                range=yr,
                showbackground=False,
                showgrid=True,
                gridcolor="rgba(120,120,120,0.35)",
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor="rgba(80,80,80,0.8)",
                tickvals=yticks,
                tickfont=dict(size=18),
            ),
            zaxis=dict(
                title=dict(text="z (m)", font=dict(size=28)),
                range=zr,
                showbackground=False,
                showgrid=True,
                gridcolor="rgba(120,120,120,0.35)",
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor="rgba(80,80,80,0.8)",
                tickvals=zticks,
                tickfont=dict(size=18),
            ),
            aspectmode="cube",
            camera=dict(eye=dict(x=-1.8, y=-1.8, z=0.9)),
        ),
        margin=dict(l=20, r=20, t=70, b=20),
        width=1000,
        height=760,
        annotations=[
            dict(
                x=0.90,
                y=0.50,
                xref="paper",
                yref="paper",
                text="|B| (nT)",
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                font=dict(size=18)
            )
        ],
    )

    out_html = os.path.join(session_dir, "cones_raw_manual.html")
    fig.write_html(out_html)

    try:
        out_png = os.path.join(session_dir, "cones_raw_manual.png")
        fig.write_image(out_png, width=1200, height=800, scale=2)
        png_note = f" + {os.path.basename(out_png)}"
    except Exception:
        png_note = ""

    fig.show()
    return out_html, png_note

def plot_cones_corrected_plotly(session_dir: str, x, y, z, bx, by, bz, bmag,
                                title: str = "Corrected field map (cones)",
                                cmin: float = 0.0,
                                cmax: float = None,
                                sizemode: str = "scaled",
                                sizeref: float = 0.6,
                                basename: str = "cones_corr_replot"):
    """
    Plotly cone plot in the same style as the GUI/raw cone export,
    using per-map cone scaling (intuitive within one map),
    while keeping a fixed absolute color scale.
    """
    if not PLOTLY_OK:
        raise RuntimeError(f"Plotly not available: {PLOTLY_ERR}")

    if cmax is None:
        cmax = float(np.nanmax(bmag)) if np.isfinite(np.nanmax(bmag)) else 1.0
    if cmax <= cmin:
        cmax = cmin + 1.0

    pad = 0.05
    xr = [float(np.min(x) - pad), float(np.max(x) + pad)]
    yr = [float(np.min(y) - pad), float(np.max(y) + pad)]
    zr = [float(np.min(z) - pad), float(np.max(z) + pad)]

    xticks = sorted(np.unique(np.round(x, 10)).tolist())
    yticks = sorted(np.unique(np.round(y, 10)).tolist())
    zticks = sorted(np.unique(np.round(z, 10)).tolist())

    fig = go.Figure()

    grid_color = "rgba(120,120,120,0.55)"
    grid_width = 2

    for yy in yticks:
        for zz in zticks:
            fig.add_trace(go.Scatter3d(
                x=[xticks[0], xticks[-1]],
                y=[yy, yy],
                z=[zz, zz],
                mode="lines",
                line=dict(color=grid_color, width=grid_width),
                hoverinfo="skip",
                showlegend=False
            ))

    for xx in xticks:
        for zz in zticks:
            fig.add_trace(go.Scatter3d(
                x=[xx, xx],
                y=[yticks[0], yticks[-1]],
                z=[zz, zz],
                mode="lines",
                line=dict(color=grid_color, width=grid_width),
                hoverinfo="skip",
                showlegend=False
            ))

    for xx in xticks:
        for yy in yticks:
            fig.add_trace(go.Scatter3d(
                x=[xx, xx],
                y=[yy, yy],
                z=[zticks[0], zticks[-1]],
                mode="lines",
                line=dict(color=grid_color, width=grid_width),
                hoverinfo="skip",
                showlegend=False
            ))

    fig.add_trace(go.Cone(
        x=x, y=y, z=z,
        u=bx, v=by, w=bz,
        colorscale="Jet",
        cmin=cmin, cmax=cmax,
        sizemode=sizemode,
        sizeref=sizeref,
        anchor="cm",
        showscale=True,
        colorbar=dict(
            title=None,
            thickness=26,
            len=0.74,
            x=0.87,
            y=0.50,
            tickfont=dict(size=20, family="STIX Two Text, STIXGeneral, Cambria, Times New Roman, serif"),
        )
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=30,
                family="STIX Two Text, STIXGeneral, Cambria, Times New Roman, serif"
            ),
            x=0.06,
            xanchor="left"
        ),
        font=dict(
            family="STIX Two Text, STIXGeneral, Cambria, Times New Roman, serif",
            size=22
        ),
        scene=dict(
            xaxis=dict(
                title=dict(
                    text="x (m)",
                    font=dict(size=26, family="STIX Two Text, STIXGeneral, Cambria, Times New Roman, serif")
                ),
                range=xr,
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor="rgba(80,80,80,1.0)",
                tickvals=xticks,
                tickfont=dict(size=16, family="STIX Two Text, STIXGeneral, Cambria, Times New Roman, serif"),
            ),
            yaxis=dict(
                title=dict(
                    text="y (m)",
                    font=dict(size=26, family="STIX Two Text, STIXGeneral, Cambria, Times New Roman, serif")
                ),
                range=yr,
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor="rgba(80,80,80,1.0)",
                tickvals=yticks,
                tickfont=dict(size=16, family="STIX Two Text, STIXGeneral, Cambria, Times New Roman, serif"),
            ),
            zaxis=dict(
                title=dict(
                    text="z (m)",
                    font=dict(size=26, family="STIX Two Text, STIXGeneral, Cambria, Times New Roman, serif")
                ),
                range=zr,
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor="rgba(80,80,80,1.0)",
                tickvals=zticks,
                tickfont=dict(size=16, family="STIX Two Text, STIXGeneral, Cambria, Times New Roman, serif"),
            ),
            aspectmode="cube",
            camera=dict(eye=dict(x=-1.8, y=-1.8, z=0.9)),
        ),
        margin=dict(l=10, r=10, t=70, b=10),
        width=1100,
        height=820,
        annotations=[
            dict(
                x=0.870,
                y=0.50,
                xref="paper",
                yref="paper",
                text="|B| (nT)",
                showarrow=False,
                textangle=0,
                xanchor="right",
                yanchor="middle",
                font=dict(
                    size=26,
                    family="STIX Two Text, STIXGeneral, Cambria, Times New Roman, serif"
                )
            )
        ],
    )

    out_html = os.path.join(session_dir, f"{basename}.html")
    fig.write_html(out_html)

    try:
        out_png = os.path.join(session_dir, f"{basename}.png")
        fig.write_image(out_png, width=1400, height=1000, scale=2)
        png_note = f" + {os.path.basename(out_png)}"
    except Exception:
        png_note = ""

    fig.show()
    return out_html, png_note

def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)

def parseval_psd_check(x: np.ndarray, f: np.ndarray, pxx: np.ndarray) -> dict:
    """
    Compare time-domain variance with integral of one-sided PSD.
    x: the exact vector used for PSD (after detrend etc.)
    f, pxx: one-sided frequency vector and PSD (units^2/Hz)

    Returns dict with var_time, var_psd, ratio, err_pct.
    """
    x = np.asarray(x, dtype=np.float64)
    x0 = x - np.mean(x)               
    var_time = float(np.mean(x0**2))  
    var_psd = float(_trapz(pxx, f))  
    ratio = var_psd / var_time if var_time > 0 else float("nan")
    err_pct = 100.0 * (ratio - 1.0) if np.isfinite(ratio) else float("nan")
    return {
        "var_time": var_time,
        "var_psd": var_psd,
        "ratio": ratio,
        "err_pct": err_pct,
    }


def ask(prompt, default=None, valid=None):
    if default is not None:
        prompt = f"{prompt} [{default}]"
    prompt = prompt + ": "
    while True:
        s = input(prompt).strip()
        if not s and default is not None:
            s = str(default)
        if valid is None:
            return s
        if s.lower() in [v.lower() for v in valid]:
            return s
        print(f"Please choose one of: {', '.join(valid)}")


def ask_yesno(prompt, default="n"):
    return ask(prompt + " (y/n)", default=default, valid=["y", "n"]).lower() == "y"


def ask_int(prompt, default=None, min_val=None, max_val=None):
    while True:
        s = ask(prompt, default=default)
        try:
            v = int(s)
        except ValueError:
            print("Please enter an integer.")
            continue
        if min_val is not None and v < min_val:
            print(f"Must be >= {min_val}")
            continue
        if max_val is not None and v > max_val:
            print(f"Must be <= {max_val}")
            continue
        return v


def ask_float(prompt, default=None, min_val=None, max_val=None):
    while True:
        s = ask(prompt, default=default)
        try:
            v = float(s)
        except ValueError:
            print("Please enter a number.")
            continue
        if min_val is not None and v < min_val:
            print(f"Must be >= {min_val}")
            continue
        if max_val is not None and v > max_val:
            print(f"Must be <= {max_val}")
            continue
        return v


def pick_folder_dialog(initial_dir=None) -> Optional[str]:
    if not TK_OK:
        return None
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Select FieldMap session folder (map_YYYYMMDD_HHMMSS)",
        initialdir=initial_dir or os.getcwd(),
    )
    root.destroy()
    return folder if folder else None


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def median(values: List[float]) -> float:
    a = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return float("nan")
    return float(np.median(a))


# ============================================================
# Session loading (config, fs, stream, summary, offsets)
# ============================================================

def infer_fs(session_dir: str, cfg: dict) -> float:
    idx_path = os.path.join(session_dir, "stream_index.csv")
    if os.path.exists(idx_path):
        try:
            rows = read_csv_rows(idx_path)
            vals = []
            for row in rows:
                if "actual_fs" in row:
                    try:
                        vals.append(float(row["actual_fs"]))
                    except Exception:
                        pass
            fs = median(vals)
            if np.isfinite(fs) and fs > 0:
                return fs
        except Exception:
            pass

    fs_req = cfg.get("fs_req", None)
    if fs_req is None:
        raise RuntimeError("Could not infer fs: no stream_index.csv and config.json missing fs_req.")
    return float(fs_req)


def memmap_stream_xyz(session_dir: str):
    bin_path = os.path.join(session_dir, "stream_raw_f32.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Missing: {bin_path}")
    mm = np.memmap(bin_path, dtype=np.float32, mode="r")
    n3 = mm.shape[0]
    n = n3 // 3
    if n <= 0:
        raise RuntimeError("stream_raw_f32.bin appears empty.")
    mm = mm[: n * 3]
    xyz = mm.reshape(n, 3) 
    return xyz, n

def memmap_stream_aux(session_dir: str):
    """
    AUX file is float32 interleaved per sample:
      AUX1, AUX2, AUX3, AUX1, AUX2, AUX3, ...
    Returns:
      aux_mm shaped (N,3), n_samples
    """
    bin_path = os.path.join(session_dir, "stream_aux_f32.bin")
    if not os.path.exists(bin_path):
        return None, 0

    mm = np.memmap(bin_path, dtype=np.float32, mode="r")
    n3 = mm.shape[0]
    n = n3 // 3
    if n <= 0:
        return None, 0

    mm = mm[: n * 3]
    aux = mm.reshape(n, 3)
    return aux, n


def get_segment_aux_for_timeplot(st: "State") -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (t, seg_v) for AUX time trace.
    AUX is always raw volts, no offset correction.
    """
    if st.aux_mm is None or st.aux_n_total <= 0:
        raise RuntimeError("No stream_aux_f32.bin found in this session.")

    s0, s1 = st.sel.s0, st.sel.s1
    s0 = max(0, min(st.aux_n_total, s0))
    s1 = max(0, min(st.aux_n_total, s1))
    n = s1 - s0
    if n <= 0:
        raise RuntimeError("Empty AUX selection.")

    _, step = downsample_slice(n, st.set.max_plot_points)
    seg_v = np.asarray(st.aux_mm[s0:s1:step], dtype=np.float32)
    t = (np.arange(seg_v.shape[0], dtype=float) * step + s0) / st.fs
    return t, seg_v


def single_sided_fft_amplitude(x: np.ndarray, fs: float, window: str = "hann", detrend: str = "mean"):
    """
    Single-sided amplitude spectrum in the same units as x (here: V).
    Uses coherent-gain correction so a pure sine shows roughly its amplitude.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        raise ValueError("Signal too short for FFT.")

    if detrend == "linear":
        x = detrend_linear(x)
    else:
        x = x - np.mean(x)

    if window.lower() == "hann":
        w = np.hanning(n)
    elif window.lower() == "hamming":
        w = np.hamming(n)
    else:
        w = np.ones(n, dtype=np.float64)

    cg = np.sum(w) / n
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(n, d=1.0 / fs)

    amp = np.abs(X) / (n * cg)

    if n % 2 == 0:
        if len(amp) > 2:
            amp[1:-1] *= 2.0
    else:
        if len(amp) > 1:
            amp[1:] *= 2.0

    return f, amp

def estimate_period_samples_from_fft(x: np.ndarray, fs: float) -> Optional[int]:
    """
    Estimate dominant period in samples from the FFT peak.
    Works better than zero crossings when the waveform is noisy or offset.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 32:
        return None

    x0 = x - np.mean(x)

    w = np.hanning(n)
    X = np.fft.rfft(x0 * w)
    f = np.fft.rfftfreq(n, d=1.0 / fs)

    mag = np.abs(X)
    if len(mag) < 3:
        return None

    mag[0] = 0.0

    k = int(np.argmax(mag))
    if k <= 0 or k >= len(f):
        return None

    f0 = float(f[k])
    if not np.isfinite(f0) or f0 <= 0:
        return None

    period_samples = int(round(fs / f0))
    if period_samples < 2:
        return None

    return period_samples


def find_signal_onset_by_envelope(x: np.ndarray, sigma_thresh: float = 5.0, smooth: int = 200) -> int:
    """
    Find the onset of a burst-like signal using an absolute-value envelope.
    Uses the first 10% of the signal as a noise baseline.
    Returns a sample index.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 20:
        return 0

    x0 = x - np.mean(x)
    env = np.abs(x0)

    smooth = max(5, int(smooth))
    if smooth >= n:
        smooth = max(5, n // 10)
    ker = np.ones(smooth, dtype=np.float64) / smooth
    env_s = np.convolve(env, ker, mode="same")

    n0 = max(20, int(0.1 * n))
    base = env_s[:n0]
    mu = float(np.mean(base))
    sd = float(np.std(base, ddof=1)) if len(base) > 1 else 0.0
    thr = mu + sigma_thresh * sd

    idx = np.where(env_s > thr)[0]
    if len(idx) == 0:
        return 0

    return int(idx[0])

def get_cone_cscale_limits(st: "State", bmag: np.ndarray):
    """
    Returns (cmin, cmax) for cone plots in nT.
    """
    cmin = float(st.set.cone_cmin_nt)

    if st.set.cone_cmax_mode.lower() == "auto":
        cmax = float(np.nanmax(bmag)) if np.size(bmag) else 1.0
        if not np.isfinite(cmax) or cmax <= cmin:
            cmax = cmin + 1.0
    else:
        cmax = float(st.set.cone_cmax_fixed_nt)
        if cmax <= cmin:
            cmax = cmin + 1.0

    return cmin, cmax


def load_summary(session_dir: str):
    for name in ["summary.csv", "summary_raw.csv"]:
        p = os.path.join(session_dir, name)
        if os.path.exists(p):
            rows = read_csv_rows(p)
            return rows, name
    return None, None

def load_aux_points(session_dir: str) -> List[Dict[str, str]]:
    """
    Load AUX points recorded by the FieldMap GUI from aux_points.csv.
    Returns an empty list if the file does not exist.
    """
    p = os.path.join(session_dir, "aux_points.csv")
    if not os.path.exists(p):
        return []
    rows = read_csv_rows(p)
    return rows

def pick_second_session_folder(title: str = "Select second fieldmap folder") -> Optional[str]:
    """
    Ask for a second session folder using a dialog.
    Falls back to manual text input if tkinter is unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title=title)
        root.destroy()
        return folder if folder else None
    except Exception:
        s = input(f"{title} (enter folder path, blank=cancel): ").strip()
        return s if s else None


def _first_present_value(row: Dict[str, str], candidates: List[str]) -> float:
    for k in candidates:
        if k in row and row[k] not in ("", None):
            return float(row[k])
    raise KeyError(f"None of the expected columns found: {candidates}")


def load_corrected_fieldmap_from_summary(session_dir: str, cfg: dict):
    """
    Loads a COMPLETE corrected fieldmap from summary.csv and validates that:
      - corrected summary exists
      - START and END offsets exist in map_stats.json
      - expected number of points is present (default 5^3 = 125)

    Returns:
      dict with arrays:
        ix, iy, iz, x, y, z,
        bx_corr_nt, by_corr_nt, bz_corr_nt, bmag_nt,
        off0_nt, off1_nt
    """
    summary_path = os.path.join(session_dir, "summary.csv")
    if not os.path.exists(summary_path):
        raise RuntimeError("Fieldmap replot / QA requires summary.csv (corrected map). summary_raw.csv is not sufficient.")

    rows = read_csv_rows(summary_path)
    if len(rows) == 0:
        raise RuntimeError("summary.csv is empty.")

    ms_path = os.path.join(session_dir, "map_stats.json")
    if not os.path.exists(ms_path):
        raise RuntimeError("Fieldmap replot / QA requires map_stats.json.")

    ms = load_json(ms_path)

    if "off0_offset_v" not in ms or "off1_offset_v" not in ms:
        raise RuntimeError("Fieldmap replot / QA requires both START and END offsets in map_stats.json.")

    grid_n = int(cfg.get("grid_n", 5))
    n_expected = grid_n ** 3

    points_captured = int(ms.get("points_captured", len(rows)))
    points_expected = int(ms.get("points_expected", n_expected))
    run_complete = bool(ms.get("run_complete", len(rows) == n_expected))

    if not run_complete:
        raise RuntimeError("Fieldmap replot / QA requires a complete run (run_complete=true).")
    if points_captured != n_expected or points_expected != n_expected or len(rows) != n_expected:
        raise RuntimeError(
            f"Fieldmap replot / QA requires a complete {grid_n}x{grid_n}x{grid_n} map "
            f"({n_expected} points). Found len(summary)={len(rows)}, "
            f"points_captured={points_captured}, points_expected={points_expected}."
        )

    ixs, iys, izs = [], [], []
    xs, ys, zs = [], [], []
    bx_corr, by_corr, bz_corr, bmag = [], [], [], []

    for r in rows:
        ixs.append(int(float(r["ix"])))
        iys.append(int(float(r["iy"])))
        izs.append(int(float(r["iz"])))

        xs.append(float(r["x"]))
        ys.append(float(r["y"]))
        zs.append(float(r["z"]))

        bx_corr.append(float(r["Bx_corr_nT"]))
        by_corr.append(float(r["By_corr_nT"]))
        bz_corr.append(float(r["Bz_corr_nT"]))
        bmag.append(float(r["Bmag_corr_nT"]))

    scale = np.asarray(
        ms.get("scale_nt_per_v", cfg.get("scale_nt_per_v", [7000.0, 7000.0, 7000.0])),
        dtype=float
    ).reshape(3,)

    off0_v = np.asarray(ms["off0_offset_v"], dtype=float).reshape(3,)
    off1_v = np.asarray(ms["off1_offset_v"], dtype=float).reshape(3,)
    off0_nt = off0_v * scale
    off1_nt = off1_v * scale

    return {
        "ix": np.asarray(ixs, dtype=int),
        "iy": np.asarray(iys, dtype=int),
        "iz": np.asarray(izs, dtype=int),
        "x": np.asarray(xs, dtype=float),
        "y": np.asarray(ys, dtype=float),
        "z": np.asarray(zs, dtype=float),
        "bx_corr_nt": np.asarray(bx_corr, dtype=float),
        "by_corr_nt": np.asarray(by_corr, dtype=float),
        "bz_corr_nt": np.asarray(bz_corr, dtype=float),
        "bmag_nt": np.asarray(bmag, dtype=float),
        "grid_n": grid_n,
        "off0_nt": off0_nt,
        "off1_nt": off1_nt,
    }


def compute_fieldmap_metrics(fm: dict):
    """
    Computes fieldmap QA metrics.

    Reports BOTH:
      - adjacent-point field differences (nT)
      - corresponding adjacent-point gradients (nT/m)

    RMS metrics follow the definitions:
      ΔB_RMS = sqrt( (1/N_g) * sum_i sum_k (ΔB_i(r_k))^2 )
      G_RMS  = sqrt( (1/N_g) * sum_i sum_k (g_i(r_k))^2 )
    """
    ix = fm["ix"]
    iy = fm["iy"]
    iz = fm["iz"]
    x = fm["x"]
    y = fm["y"]
    z = fm["z"]
    b = fm["bmag_nt"]

    ux = np.array(sorted(np.unique(ix)))
    uy = np.array(sorted(np.unique(iy)))
    uz = np.array(sorted(np.unique(iz)))

    if len(ux) < 2 or len(uy) < 2 or len(uz) < 2:
        raise RuntimeError("Need at least 2 grid points per axis for gradient computation.")

    x_by_ix = {i: float(np.mean(x[ix == i])) for i in ux}
    y_by_iy = {j: float(np.mean(y[iy == j])) for j in uy}
    z_by_iz = {k: float(np.mean(z[iz == k])) for k in uz}

    dx_vals = np.diff([x_by_ix[i] for i in ux])
    dy_vals = np.diff([y_by_iy[j] for j in uy])
    dz_vals = np.diff([z_by_iz[k] for k in uz])

    dx = float(np.median(np.abs(dx_vals)))
    dy = float(np.median(np.abs(dy_vals)))
    dz = float(np.median(np.abs(dz_vals)))

    if dx <= 0 or dy <= 0 or dz <= 0:
        raise RuntimeError("Invalid grid spacing derived from summary.csv coordinates.")

    by_key = {(int(a), int(b_), int(c)): float(v) for a, b_, c, v in zip(ix, iy, iz, b)}

    cx = int(ux[len(ux) // 2])
    cy = int(uy[len(uy) // 2])
    cz = int(uz[len(uz) // 2])

    if (cx, cy, cz) not in by_key:
        raise RuntimeError(f"Center point ({cx},{cy},{cz}) not found in fieldmap.")

    B_center = by_key[(cx, cy, cz)]

    i_min = int(np.argmin(b))
    i_max = int(np.argmax(b))

    B_min = float(b[i_min])
    B_max = float(b[i_max])
    dB = B_max - B_min
    B_mean_V = float(np.mean(b))
    B_std_V = float(np.std(b, ddof=1))

    pt_min = (int(ix[i_min]), int(iy[i_min]), int(iz[i_min]))
    pt_max = (int(ix[i_max]), int(iy[i_max]), int(iz[i_max]))

    next_x = {ux[i]: ux[i + 1] for i in range(len(ux) - 1)}
    next_y = {uy[j]: uy[j + 1] for j in range(len(uy) - 1)}
    next_z = {uz[k]: uz[k + 1] for k in range(len(uz) - 1)}

    dBx_all = []
    dBy_all = []
    dBz_all = []

    gx_all = []
    gy_all = []
    gz_all = []

    for i0 in ux[:-1]:
        for j0 in uy:
            for k0 in uz:
                p = (int(i0), int(j0), int(k0))
                px = (int(next_x[i0]), int(j0), int(k0))
                if p in by_key and px in by_key:
                    dBx = abs(by_key[px] - by_key[p])
                    dBx_all.append(dBx)
                    gx_all.append(dBx / dx)

    for i0 in ux:
        for j0 in uy[:-1]:
            for k0 in uz:
                p = (int(i0), int(j0), int(k0))
                py = (int(i0), int(next_y[j0]), int(k0))
                if p in by_key and py in by_key:
                    dBy = abs(by_key[py] - by_key[p])
                    dBy_all.append(dBy)
                    gy_all.append(dBy / dy)

    for i0 in ux:
        for j0 in uy:
            for k0 in uz[:-1]:
                p = (int(i0), int(j0), int(k0))
                pz = (int(i0), int(j0), int(next_z[k0]))
                if p in by_key and pz in by_key:
                    dBz = abs(by_key[pz] - by_key[p])
                    dBz_all.append(dBz)
                    gz_all.append(dBz / dz)

    dBx_adj_max = float(np.max(dBx_all)) if dBx_all else float("nan")
    dBy_adj_max = float(np.max(dBy_all)) if dBy_all else float("nan")
    dBz_adj_max = float(np.max(dBz_all)) if dBz_all else float("nan")

    Gx_max = float(np.max(gx_all)) if gx_all else float("nan")
    Gy_max = float(np.max(gy_all)) if gy_all else float("nan")
    Gz_max = float(np.max(gz_all)) if gz_all else float("nan")

    N_gx = len(dBx_all)
    N_gy = len(dBy_all)
    N_gz = len(dBz_all)
    N_g = N_gx + N_gy + N_gz

    if N_g > 0:
        dB_RMS_adj = float(np.sqrt(
            (np.sum(np.square(dBx_all)) +
             np.sum(np.square(dBy_all)) +
             np.sum(np.square(dBz_all))) / N_g
        ))
        G_RMS = float(np.sqrt(
            (np.sum(np.square(gx_all)) +
             np.sum(np.square(gy_all)) +
             np.sum(np.square(gz_all))) / N_g
        ))
    else:
        dB_RMS_adj = float("nan")
        G_RMS = float("nan")

    return {
        "B_center_nT": float(B_center),
        "B_center_point": (cx, cy, cz),
        "B_min_nT": B_min,
        "B_min_point": pt_min,
        "B_max_nT": B_max,
        "B_max_point": pt_max,
        "dB_nT": float(dB),
        "B_mean_V_nT": B_mean_V,
        "B_std_V_nT": B_std_V,

        "dBx_adj_max_nT": dBx_adj_max,
        "dBy_adj_max_nT": dBy_adj_max,
        "dBz_adj_max_nT": dBz_adj_max,

        "Gx_max_nT_per_m": Gx_max,
        "Gy_max_nT_per_m": Gy_max,
        "Gz_max_nT_per_m": Gz_max,

        "dB_RMS_adj_nT": dB_RMS_adj,
        "G_RMS_nT_per_m": G_RMS,

        "N_gx": N_gx,
        "N_gy": N_gy,
        "N_gz": N_gz,
        "N_g": N_g,

        "dx_m": dx,
        "dy_m": dy,
        "dz_m": dz,

        "off0_nt": fm.get("off0_nt"),
        "off1_nt": fm.get("off1_nt"),
        "doff_nt": None if ("off0_nt" not in fm or "off1_nt" not in fm) else (fm["off1_nt"] - fm["off0_nt"]),
    }

def compute_fieldmap_mean_and_diff(fm_a: dict, fm_b: dict):
    """
    Compute:
      - mean fieldmap: (fm_a + fm_b)/2
      - half-difference fieldmap: (fm_a - fm_b)/2

    Returns:
      fm_mean, fm_diff
    """
    for key in ["ix", "iy", "iz"]:
        if not np.array_equal(fm_a[key], fm_b[key]):
            raise RuntimeError(f"Mismatch in {key} indices.")

    for key in ["x", "y", "z"]:
        if not np.allclose(fm_a[key], fm_b[key], atol=1e-12):
            raise RuntimeError(f"Mismatch in {key} coordinates.")

    bx_mean = 0.5 * (fm_a["bx_corr_nt"] + fm_b["bx_corr_nt"])
    by_mean = 0.5 * (fm_a["by_corr_nt"] + fm_b["by_corr_nt"])
    bz_mean = 0.5 * (fm_a["bz_corr_nt"] + fm_b["bz_corr_nt"])
    bmag_mean = np.sqrt(bx_mean**2 + by_mean**2 + bz_mean**2)

    bx_diff = 0.5 * (fm_a["bx_corr_nt"] - fm_b["bx_corr_nt"])
    by_diff = 0.5 * (fm_a["by_corr_nt"] - fm_b["by_corr_nt"])
    bz_diff = 0.5 * (fm_a["bz_corr_nt"] - fm_b["bz_corr_nt"])
    bmag_diff = np.sqrt(bx_diff**2 + by_diff**2 + bz_diff**2)

    base = {
        "ix": fm_a["ix"],
        "iy": fm_a["iy"],
        "iz": fm_a["iz"],
        "x": fm_a["x"],
        "y": fm_a["y"],
        "z": fm_a["z"],
    }

    fm_mean = {
        **base,
        "bx_corr_nt": bx_mean,
        "by_corr_nt": by_mean,
        "bz_corr_nt": bz_mean,
        "bmag_nt": bmag_mean,
    }

    fm_diff = {
        **base,
        "bx_corr_nt": bx_diff,
        "by_corr_nt": by_diff,
        "bz_corr_nt": bz_diff,
        "bmag_nt": bmag_diff,
    }

    return fm_mean, fm_diff

def replot_corrected_fieldmap_3d(session_dir: str, fm: dict):
    """
    Replot corrected 3D fieldmap in a cleaner thesis-style matplotlib figure.
    Saves PNG and PDF and opens the figure.
    """
    x = fm["x"]
    y = fm["y"]
    z = fm["z"]
    u = fm["bx_corr_nt"]
    v = fm["by_corr_nt"]
    w = fm["bz_corr_nt"]
    bmag = fm["bmag_nt"]

    rc = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    }

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.2, 6.2))
        ax = fig.add_subplot(111, projection="3d")

        norm = plt.Normalize(float(np.min(bmag)), float(np.max(bmag)))
        cmap = plt.cm.viridis
        colors = cmap(norm(bmag))

        ux = np.unique(np.round(x, 12))
        uy = np.unique(np.round(y, 12))
        uz = np.unique(np.round(z, 12))
        dx = np.median(np.diff(np.sort(ux))) if len(ux) > 1 else 0.25
        dy = np.median(np.diff(np.sort(uy))) if len(uy) > 1 else 0.25
        dz = np.median(np.diff(np.sort(uz))) if len(uz) > 1 else 0.25
        spacing = float(np.median([dx, dy, dz]))

        max_mag = float(np.max(bmag)) if np.max(bmag) > 0 else 1.0
        scale = 0.35 * spacing / max_mag

        ax.quiver(
            x, y, z,
            u * scale, v * scale, w * scale,
            colors=colors,
            linewidths=1.0,
            arrow_length_ratio=0.25,
            normalize=False
        )

        ax.scatter(x, y, z, c=colors, s=12, depthshade=False)

        ax.set_xlabel(r"$x$ (m)", labelpad=8)
        ax.set_ylabel(r"$y$ (m)", labelpad=8)
        ax.set_zlabel(r"$z$ (m)", labelpad=8)
        ax.set_title(r"Corrected 3D field map", pad=12)

        xr = np.ptp(x)
        yr = np.ptp(y)
        zr = np.ptp(z)
        ax.set_box_aspect((xr if xr > 0 else 1, yr if yr > 0 else 1, zr if zr > 0 else 1))

        ax.view_init(elev=22, azim=-55)

        try:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("0.75")
            ax.yaxis.pane.set_edgecolor("0.75")
            ax.zaxis.pane.set_edgecolor("0.75")
        except Exception:
            pass
        ax.grid(False)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.06)
        cbar.set_label(r"$|\overline{B}|$ (nT)")

        fig.tight_layout()

        out_png = os.path.join(session_dir, "fieldmap_corr_replot.png")
        out_pdf = os.path.join(session_dir, "fieldmap_corr_replot.pdf")
        fig.savefig(out_png, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")

        plt.show()

    return out_png, out_pdf

def get_recorded_point_windows(st: "State") -> List[Tuple[int, int]]:
    """
    Returns [(start_idx, end_idx), ...] for recorded map points, but only if:
      - a corrected/summary file is available
      - the run is complete (full fieldmap)
    Otherwise returns [].

    The windows are in raw sample indices.
    """
    if not st.summary_rows:
        return []

    ms_path = os.path.join(st.session_dir, "map_stats.json")
    if not os.path.exists(ms_path):
        return []

    try:
        ms = load_json(ms_path)
    except Exception:
        return []

    if not bool(ms.get("run_complete", False)):
        return []

    pts_cap = int(ms.get("points_captured", 0))
    pts_exp = int(ms.get("points_expected", 0))
    if pts_cap <= 0 or pts_exp <= 0 or pts_cap != pts_exp:
        return []

    windows = []
    for r in st.summary_rows:
        try:
            s0 = int(float(r["start_idx"]))
            s1 = int(float(r["end_idx"]))
            if s1 > s0:
                windows.append((s0, s1))
        except Exception:
            continue

    if len(windows) != pts_exp:
        return []

    return windows

def overlay_point_windows(ax: plt.Axes, st: "State", t0_s: float, t1_s: float):
    """
    Overlay recorded point windows as visible spans + boundary lines
    on a time-trace axis.
    Only windows overlapping the currently displayed interval are drawn.
    """
    windows = get_recorded_point_windows(st)
    if not windows:
        return

    fs = float(st.fs)

    for s0, s1 in windows:
        a = s0 / fs
        b = s1 / fs

        if b < t0_s or a > t1_s:
            continue

        aa = max(a, t0_s)
        bb = min(b, t1_s)

        ax.axvspan(aa, bb, alpha=0.18, color="0.75", lw=0, zorder=0)

        ax.axvline(aa, color="0.35", linestyle="--", linewidth=0.8, alpha=0.9, zorder=1)
        ax.axvline(bb, color="0.35", linestyle="--", linewidth=0.8, alpha=0.9, zorder=1)


def find_point_block(summary_rows: List[Dict[str, str]], ix: int, iy: int, iz: int):
    for row in summary_rows:
        try:
            if int(row["ix"]) == ix and int(row["iy"]) == iy and int(row["iz"]) == iz:
                s0 = int(float(row["start_idx"]))
                s1 = int(float(row["end_idx"]))
                tmid = None
                if "tmid_s" in row:
                    try:
                        tmid = float(row["tmid_s"])
                    except Exception:
                        tmid = None
                return s0, s1, tmid, row
        except Exception:
            continue
    raise ValueError(f"Point ({ix},{iy},{iz}) not found in summary.")


def load_scale_nt_per_v(cfg: dict) -> np.ndarray:
    s = cfg.get("scale_nt_per_v", [1.0, 1.0, 1.0])
    return np.asarray(s, dtype=float).reshape(3,)


def load_offsets(session_dir: str):
    """
    Supports:
    - flat keys: off0_tmid_s/off0_offset_v etc
    - nested: off0:{tmid_s,offset_v}
    """
    ms_path = os.path.join(session_dir, "map_stats.json")
    if not os.path.exists(ms_path):
        return None, None
    try:
        ms = load_json(ms_path)
    except Exception:
        return None, None

    off0 = off1 = None

    if ("off0_tmid_s" in ms) and ("off0_offset_v" in ms):
        off0 = {"tmid_s": float(ms["off0_tmid_s"]),
                "offset_v": np.asarray(ms["off0_offset_v"], dtype=float).reshape(3,)}
    if ("off1_tmid_s" in ms) and ("off1_offset_v" in ms):
        off1 = {"tmid_s": float(ms["off1_tmid_s"]),
                "offset_v": np.asarray(ms["off1_offset_v"], dtype=float).reshape(3,)}

    def _nested(key):
        if key not in ms or not isinstance(ms[key], dict):
            return None
        obj = ms[key]
        if "tmid_s" not in obj or "offset_v" not in obj:
            return None
        return {"tmid_s": float(obj["tmid_s"]),
                "offset_v": np.asarray(obj["offset_v"], dtype=float).reshape(3,)}

    if off0 is None:
        off0 = _nested("off0")
    if off1 is None:
        off1 = _nested("off1")

    return off0, off1


def offset_at_time(t_s: np.ndarray, off0, off1):
    if off0 is None and off1 is None:
        return None
    if off0 is None and off1 is not None:
        return np.repeat(off1["offset_v"][None, :], repeats=len(t_s), axis=0)
    if off1 is None and off0 is not None:
        return np.repeat(off0["offset_v"][None, :], repeats=len(t_s), axis=0)

    t0 = float(off0["tmid_s"])
    t1 = float(off1["tmid_s"])
    v0 = np.asarray(off0["offset_v"], dtype=float).reshape(3,)
    v1 = np.asarray(off1["offset_v"], dtype=float).reshape(3,)

    if t1 <= t0 + 1e-12:
        return np.repeat(v0[None, :], repeats=len(t_s), axis=0)

    a = (t_s - t0) / (t1 - t0)
    a = np.clip(a, 0.0, 1.0)
    return v0[None, :] + a[:, None] * (v1[None, :] - v0[None, :])


def convert_units_from_volts(data_v: np.ndarray, units: str, scale_nt_per_v: np.ndarray):
    units = units.lower()
    if units == "v":
        return data_v.astype(np.float32, copy=False), "V"

    data_nt = data_v.astype(np.float64, copy=False) * scale_nt_per_v.reshape(1, 3)
    if units == "nt":
        return data_nt.astype(np.float32, copy=False), "nT"
    if units == "ut":
        return (data_nt / 1000.0).astype(np.float32, copy=False), "µT"
    if units == "t":
        return (data_nt * 1e-9).astype(np.float32, copy=False), "T"

    raise ValueError("units must be one of: V, nT, uT, T")


def apply_offset(seg_v: np.ndarray,
                 fs: float,
                 s0: int,
                 off_mode: str,
                 off0,
                 off1,
                 point_tmid_s: Optional[float] = None):
    """
    seg_v: (N,3) volts
    off_mode:
      none | start | linear | pointwise
    """
    off_mode = off_mode.lower()
    if off_mode == "none":
        return seg_v

    if off_mode == "start":
        if off0 is None:
            return seg_v
        return seg_v - np.asarray(off0["offset_v"], dtype=np.float32)[None, :]

    if off_mode == "pointwise":
        if off0 is None:
            return seg_v
        if point_tmid_s is None:
            return seg_v - np.asarray(off0["offset_v"], dtype=np.float32)[None, :]
        off = offset_at_time(np.array([point_tmid_s], dtype=float), off0, off1)
        if off is None:
            return seg_v
        return seg_v - off[0].astype(np.float32)[None, :]

    if off_mode == "linear":
        if off0 is None:
            return seg_v
        n = seg_v.shape[0]
        t_s = (np.arange(n, dtype=float) + s0) / fs
        off = offset_at_time(t_s, off0, off1)
        if off is None:
            return seg_v
        return seg_v - off.astype(np.float32)

    raise ValueError("off_mode must be: none/start/linear/pointwise")


# ============================================================
# Analysis primitives (draw-only; NO plt.show() inside)
# ============================================================

def downsample_slice(n: int, max_points: int) -> Tuple[slice, int]:
    if max_points <= 0 or n <= max_points:
        return slice(0, n, 1), 1
    step = int(math.ceil(n / max_points))
    return slice(0, n, step), step


def detrend_linear(x: np.ndarray) -> np.ndarray:
    """Remove linear trend (robust for PSD near 1 Hz)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 3:
        return x - np.mean(x)
    t = np.arange(n, dtype=np.float64)
    p = np.polyfit(t, x, 1)
    trend = p[0] * t + p[1]
    return x - trend


def welch_psd(x: np.ndarray, fs: float, nperseg: int, noverlap: int, window: str = "hann"):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)

    if nperseg <= 0:
        raise ValueError("nperseg must be > 0")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
    if len(x) < nperseg:
        raise ValueError("Signal shorter than nperseg")

    step = nperseg - noverlap

    if window.lower() == "hann":
        w = np.hanning(nperseg)
    elif window.lower() == "hamming":
        w = np.hamming(nperseg)
    else:
        raise ValueError("window must be 'hann' or 'hamming'")

    u = np.sum(w * w)
    scale = 1.0 / (fs * u)
    nfft = nperseg

    kmax = (len(x) - nperseg) // step + 1
    pxx_acc = None

    for k in range(kmax):
        seg = x[k * step: k * step + nperseg]
        X = np.fft.rfft(seg * w, n=nfft)
        pxx = (np.abs(X) ** 2) * scale

        if nfft % 2 == 0:
            pxx[1:-1] *= 2.0
        else:
            pxx[1:] *= 2.0

        pxx_acc = pxx if pxx_acc is None else (pxx_acc + pxx)

    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return f, (pxx_acc / float(kmax))


def band_rms_from_psd(f: np.ndarray, pxx: np.ndarray, f_lo: float, f_hi: float) -> float:
    f = np.asarray(f, dtype=float)
    pxx = np.asarray(pxx, dtype=float)
    f_hi = min(float(f_hi), float(np.max(f)))
    if f_hi <= f_lo:
        return float("nan")
    m = (f >= f_lo) & (f <= f_hi)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(_trapz(pxx[m], f[m])))


def chunk_means(xyz_mm, s0, s1, chunk_samples: int, fs: float,
                off_mode: str, off0, off1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      t_mid (M,), mean_v (M,3) in VOLTS (after offset correction on chunk)
    """
    n = s1 - s0
    m = n // chunk_samples
    if m <= 0:
        raise ValueError("Selection shorter than one chunk.")
    t_mid = np.zeros(m, dtype=np.float64)
    mean = np.zeros((m, 3), dtype=np.float64)

    off0_vec = np.asarray(off0["offset_v"], dtype=np.float64).reshape(3,) if off0 is not None else None

    for i in range(m):
        a = s0 + i * chunk_samples
        b = a + chunk_samples
        blk = np.asarray(xyz_mm[a:b], dtype=np.float32).astype(np.float64, copy=False)

        tm = (0.5 * (a + b)) / fs
        t_mid[i] = tm

        if off_mode == "start" and off0_vec is not None:
            blk = blk - off0_vec[None, :]
        elif off_mode in ("linear", "pointwise") and off0 is not None:
            off = offset_at_time(np.array([tm], dtype=float), off0, off1)
            if off is not None:
                blk = blk - off[0][None, :]

        mean[i] = blk.mean(axis=0)

    return t_mid, mean.astype(np.float32)

def chunk_stats(xyz_mm, s0, s1, chunk_samples: int, fs: float,
                off_mode: str, off0, off1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      t_mid (M,), mean_v (M,3), std_v (M,3) in VOLTS (after offset correction per chunk)
    """
    n = s1 - s0
    m = n // chunk_samples
    if m <= 0:
        raise ValueError("Selection shorter than one chunk.")
    t_mid = np.zeros(m, dtype=np.float64)
    mean = np.zeros((m, 3), dtype=np.float64)
    std = np.zeros((m, 3), dtype=np.float64)

    off0_vec = np.asarray(off0["offset_v"], dtype=np.float64).reshape(3,) if off0 is not None else None

    for i in range(m):
        a = s0 + i * chunk_samples
        b = a + chunk_samples
        blk = np.asarray(xyz_mm[a:b], dtype=np.float32).astype(np.float64, copy=False)

        tm = (0.5 * (a + b)) / fs
        t_mid[i] = tm

        if off_mode == "start" and off0_vec is not None:
            blk = blk - off0_vec[None, :]
        elif off_mode in ("linear", "pointwise") and off0 is not None:
            off = offset_at_time(np.array([tm], dtype=float), off0, off1)
            if off is not None:
                blk = blk - off[0][None, :]

        mean[i] = blk.mean(axis=0)
        std[i] = blk.std(axis=0, ddof=1) if blk.shape[0] > 1 else 0.0

    return t_mid, mean.astype(np.float32), std.astype(np.float32)


def detect_events(t_mid: np.ndarray, y: np.ndarray, threshold: float) -> List[float]:
    dt = float(np.median(np.diff(t_mid))) if len(t_mid) > 1 else 1.0
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0
    dy = np.diff(y) / dt
    idx = np.where(np.abs(dy) >= threshold)[0] + 1
    return [float(t_mid[i]) for i in idx]


def allan_overlapping(y: np.ndarray, dt: float, taus: np.ndarray):
    y = np.asarray(y, dtype=np.float64)
    y = y - np.mean(y)
    M = len(y)
    tau_out, adev_out = [], []
    for tau in taus:
        m = int(round(tau / dt))
        if m < 1 or 2*m >= M:
            continue
        dif = y[2*m:] - 2*y[m:-m] + y[:-2*m]
        avar = 0.5 * np.mean(dif**2)
        tau_out.append(m * dt)
        adev_out.append(np.sqrt(avar))
    return np.array(tau_out), np.array(adev_out)


# ============================================================
# UI State + “analysis panels”
# ============================================================

@dataclass
class Selection:
    kind: str = "all"            
    s0: int = 0
    s1: int = 0
    label: str = ""
    point: Optional[Tuple[int, int, int]] = None
    point_tmid_s: Optional[float] = None


@dataclass
class Settings:
    units: str = "nT"
    off_mode: str = "none"       

    max_plot_points: int = 200_000
    axis_plot: str = "xyz"
    show_mag: bool = False
    show_point_windows_in_trace: bool = True

    chunk_sec: float = 1.0

    psd_axis: str = "z"
    psd_units: str = "T"       
    welch_nperseg: int = 4096
    welch_noverlap: int = 2048
    welch_window: str = "hann"
    psd_detrend: str = "linear"  
    band_lo: float = 0.1
    band_hi: float = 10.0
    asd_in_pT: bool = True

    spec_axis: str = "z"
    spec_units: str = "nT"
    spec_nfft: int = 2000
    spec_overlap: int = 1600
    spec_fmax: float = 200.0

    event_threshold_ntps: float = 5.0
    event_use_mag: bool = True

    allan_sig: str = "mag"      
    allan_tau_min: float = 1.0
    allan_tau_max: float = 600.0
    allan_tau_n: int = 30

    aux_axis_plot: str = "all"      
    aux_axis_fft: str = "all"      
    aux_fft_window: str = "hann"  
    aux_fft_detrend: str = "mean"  
    aux_fft_fmax: float = 200.0
    aux_xy_x: str = "2"         
    aux_xy_y: str = "3"        
    aux_xy_max_points: int = 50000
    aux_xy_period_source: str = "x"     
    aux_xy_periods_to_show: int = 30
    aux_xy_plot_mode: str = "line"       
    aux_xy_trigger_channel: str = "x"   
    aux_xy_trigger_sigma: float = 200.0
   
    cone_size_ref_nt: float = 12.0
    cone_sizeref: float = 0.6

    cone_cmin_nt: float = 0.0
    cone_cmax_mode: str = "fixed"  
    cone_cmax_fixed_nt: float = 12.0

    cone_sizeref: float = 0.6




@dataclass
class Panel:
    title: str
    draw: Callable[[plt.Axes], None]


@dataclass
class State:
    session_dir: str
    cfg: dict
    fs: float
    scale_nt_per_v: np.ndarray
    xyz_mm: np.memmap
    n_total: int
    off0: Optional[dict]
    off1: Optional[dict]
    summary_rows: Optional[List[Dict[str, str]]]
    summary_name: Optional[str]

    aux_mm: Optional[np.memmap] = None
    aux_n_total: int = 0
    sel: Selection = field(default_factory=Selection)
    set: Settings = field(default_factory=Settings)


def print_header(st: State):
    dur_h = st.n_total / st.fs / 3600.0
    print("\n==============================================")
    print("FieldMap QA Tool (menu)")
    print(f"Session: {st.session_dir}")
    print(f"fs≈{st.fs:.3f} Hz, samples={st.n_total:,}, duration≈{dur_h:.2f} h")
    print(f"AUX file: {'yes' if st.aux_mm is not None else 'no'}")
    print(f"scale_nt_per_v: {st.scale_nt_per_v}")
    print(f"Offsets: START={'yes' if st.off0 else 'no'}  END={'yes' if st.off1 else 'no'}")
    if st.off0:
        print(f"  off0: t={st.off0['tmid_s']:.3f}s  {st.off0['offset_v']}")
    if st.off1:
        print(f"  off1: t={st.off1['tmid_s']:.3f}s  {st.off1['offset_v']}")
    print("----------------------------------------------")
    print(f"Selection: {st.sel.label}  [{st.sel.s0}:{st.sel.s1})")
    print(f"Processing: units={st.set.units}, offset={st.set.off_mode}")
    print("==============================================\n")


# ============================================================
# Selection & settings editors
# ============================================================

def choose_selection(st: State):
    kind = ask("Select data (1=All data, 2=Specific point)", default="1", valid=["1", "2"])
    if kind == "2":
        if not st.summary_rows:
            print("[WARN] No summary.csv/summary_raw.csv found. Point selection not available.")
            return choose_selection_all(st)

        ix = ask_int("ix", default=3)
        iy = ask_int("iy", default=3)
        iz = ask_int("iz", default=3)

        s0, s1, tmid, _row = find_point_block(st.summary_rows, ix, iy, iz)
        s0 = max(0, min(st.n_total, s0))
        s1 = max(0, min(st.n_total, s1))
        st.sel = Selection(
            kind="point",
            s0=s0,
            s1=s1,
            label=f"Point ({ix},{iy},{iz}) ({st.summary_name})",
            point=(ix, iy, iz),
            point_tmid_s=tmid
        )
        if st.off0 and st.off1:
            st.set.off_mode = "pointwise"
        elif st.off0:
            st.set.off_mode = "start"
        else:
            st.set.off_mode = "none"
        return

    return choose_selection_all(st)


def choose_selection_all(st: State):
    use_window = ask_yesno("Use a time window? (recommended)", default="n")
    if use_window:
        start_s = ask_float("Start time (s)", default=0.0, min_val=0.0)
        end_s = ask_float("End time (s)", default=min(600.0, st.n_total / st.fs), min_val=0.0)
        if end_s <= start_s:
            end_s = start_s + 10.0
        s0 = int(round(start_s * st.fs))
        s1 = int(round(end_s * st.fs))
    else:
        s0, s1 = 0, st.n_total
        print("[WARN] Full stream selection can be huge. Prefer chunk-based analyses.")

    s0 = max(0, min(st.n_total, s0))
    s1 = max(0, min(st.n_total, s1))
    st.sel = Selection(kind="all", s0=s0, s1=s1,
                       label=f"All data [{s0}:{s1})", point=None, point_tmid_s=None)
    if st.off0 and st.off1:
        st.set.off_mode = "linear"
    elif st.off0:
        st.set.off_mode = "start"
    else:
        st.set.off_mode = "none"


def choose_processing(st: State):
    st.set.units = ask("Units (V/nT/uT/T)", default=st.set.units, valid=["V", "nT", "uT", "T"])
    valid = ["none"]
    if st.off0:
        valid.append("start")
        valid.append("linear")
        if st.sel.kind == "point":
            valid.append("pointwise")
    default = st.set.off_mode if st.set.off_mode in valid else "none"
    st.set.off_mode = ask("Offset mode", default=default, valid=valid)


def apply_preset(st: State) -> List[str]:
    print("\nPresets (apply settings + run bundle):")
    print("  1) Overnight QA  -> Chunk mean + Events + Allan")
    print("  2) Noise / PSD   -> PSD+ASD + Spectrogram")
    print("  3) Point debug   -> Time trace + PSD+ASD")
    p = ask("Choose preset", default="1", valid=["1", "2", "3"])

    if p == "1":
        st.set.units = "nT"
        st.set.chunk_sec = 1.0
        st.set.event_threshold_ntps = 5.0
        st.set.event_use_mag = True
        st.set.allan_sig = "mag"
        st.set.allan_tau_min = 1.0
        st.set.allan_tau_max = 600.0
        st.set.allan_tau_n = 30

        if st.off0 and st.off1:
            st.set.off_mode = "linear"
        elif st.off0:
            st.set.off_mode = "start"
        else:
            st.set.off_mode = "none"

        print("[OK] Applied Overnight QA preset.")
        return ["2", "5", "6"]

    if p == "2":
        st.set.psd_axis = "z"
        st.set.psd_units = "T"
        st.set.asd_in_pT = True
        st.set.psd_detrend = "linear"
        st.set.welch_nperseg = 4096
        st.set.welch_noverlap = 2048
        st.set.welch_window = "hann"
        st.set.band_lo = 0.8
        st.set.band_hi = 1.2

        st.set.spec_axis = "z"
        st.set.spec_units = "nT"
        st.set.spec_nfft = int(round(st.fs))       
        st.set.spec_overlap = int(round(0.8 * st.set.spec_nfft))
        st.set.spec_fmax = 200.0

        if st.off0 and st.off1:
            st.set.off_mode = "linear"
        elif st.off0:
            st.set.off_mode = "start"
        else:
            st.set.off_mode = "none"

        print("[OK] Applied Noise/PSD preset.")
        print("Tip: use a short quiet window (e.g. 5–20 min) for best PSD/spectrogram.")
        return ["3", "4"]

    st.set.units = "nT"
    st.set.axis_plot = "xyz"
    st.set.show_mag = False
    st.set.max_plot_points = 200_000

    if st.sel.kind == "point" and st.off0:
        st.set.off_mode = "pointwise" if st.off1 else "start"
    elif st.off0 and st.off1:
        st.set.off_mode = "linear"
    elif st.off0:
        st.set.off_mode = "start"
    else:
        st.set.off_mode = "none"

    st.set.psd_axis = "z"
    st.set.psd_units = "nT"
    st.set.psd_detrend = "linear"
    st.set.welch_nperseg = 4096
    st.set.welch_noverlap = 2048
    st.set.welch_window = "hann"
    st.set.band_lo = 0.8
    st.set.band_hi = 1.2

    print("[OK] Applied Point debug preset.")
    return ["1", "3"]


# ============================================================
# Panels builders (each analysis returns one or more Panel)
# ============================================================

def get_segment_volts_for_timeplot(st: State) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (t, seg_u, ylab) for time trace with downsampling, applied offset & units.
    Uses memmap slicing with step to avoid loading huge arrays.
    """
    s0, s1 = st.sel.s0, st.sel.s1
    n = s1 - s0
    sl, step = downsample_slice(n, st.set.max_plot_points)
    seg_v = np.asarray(st.xyz_mm[s0:s1:step], dtype=np.float32)
    if st.set.off_mode.lower() == "linear" and st.off0:
        idxs = (np.arange(seg_v.shape[0], dtype=float) * step + s0)
        t_s = idxs / st.fs
        off = offset_at_time(t_s, st.off0, st.off1)
        if off is not None:
            seg_v = seg_v - off.astype(np.float32)
    elif st.set.off_mode.lower() == "start" and st.off0:
        seg_v = seg_v - np.asarray(st.off0["offset_v"], dtype=np.float32)[None, :]
    elif st.set.off_mode.lower() == "pointwise" and st.off0:
        tmid = st.sel.point_tmid_s
        if tmid is not None:
            off = offset_at_time(np.array([tmid], dtype=float), st.off0, st.off1)
            if off is not None:
                seg_v = seg_v - off[0].astype(np.float32)[None, :]

    seg_u, ylab = convert_units_from_volts(seg_v, st.set.units, st.scale_nt_per_v)

    t = (np.arange(seg_u.shape[0], dtype=float) * step + s0) / st.fs
    return t, seg_u, ylab


def panels_time_trace(st: State) -> List[Panel]:
    axis = st.set.axis_plot.lower()

    def draw(ax: plt.Axes):
        t, seg_u, ylab = get_segment_volts_for_timeplot(st)
        ax.set_title(f"Time trace ({axis}, offset={st.set.off_mode})")
        ax.set_xlabel("t (s)")
        ax.set_ylabel(ylab)
        ax.grid(True)

        if axis in ("x", "y", "z"):
            idx = {"x": 0, "y": 1, "z": 2}[axis]
            ax.plot(t, seg_u[:, idx], label=axis.upper())
        else:
            ax.plot(t, seg_u[:, 0], label="X")
            ax.plot(t, seg_u[:, 1], label="Y")
            ax.plot(t, seg_u[:, 2], label="Z")

        if st.set.show_mag:
            mag = np.linalg.norm(seg_u, axis=1)
            ax.plot(t, mag, label="|B|", linewidth=2)

        if st.set.show_point_windows_in_trace:
            overlay_point_windows(ax, st, float(t[0]), float(t[-1]))

        ax.legend()

    return [Panel("Time trace", draw)]


def panels_chunk_stats(st: State) -> List[Panel]:
    chunk_samples = max(2, int(round(st.set.chunk_sec * st.fs)))
    off_mode = st.set.off_mode.lower()
    if off_mode == "pointwise":
        off_mode = "linear"

    t_mid, mean_v, std_v = chunk_stats(st.xyz_mm, st.sel.s0, st.sel.s1, chunk_samples, st.fs, off_mode, st.off0, st.off1)

    mean_u, ylab = convert_units_from_volts(mean_v, st.set.units, st.scale_nt_per_v)
    std_u, _ = convert_units_from_volts(std_v, st.set.units, st.scale_nt_per_v)

    overall_mean = np.mean(mean_u, axis=0)  
    overall_std = np.mean(std_u, axis=0)    

    print(f"[Chunk stats] overall mean ({ylab}) = {overall_mean}")
    print(f"[Chunk stats] mean chunk-std ({ylab}) = {overall_std}")
    print(f"[Chunk stats] chunk length ≈ {st.set.chunk_sec:.3g} s, chunks={len(t_mid)}")

    def draw_mean(ax: plt.Axes):
        ax.set_title(f"Chunk mean XYZ (chunk={st.set.chunk_sec:.3g}s)")
        ax.set_xlabel("t (s)")
        ax.set_ylabel(ylab)
        ax.grid(True)
        ax.plot(t_mid, mean_u[:, 0], label="X")
        ax.plot(t_mid, mean_u[:, 1], label="Y")
        ax.plot(t_mid, mean_u[:, 2], label="Z")
        ax.legend()

    def draw_std(ax: plt.Axes):
        ax.set_title(f"Chunk std XYZ (chunk={st.set.chunk_sec:.3g}s)")
        ax.set_xlabel("t (s)")
        ax.set_ylabel(ylab)
        ax.grid(True)
        ax.plot(t_mid, std_u[:, 0], label="X std")
        ax.plot(t_mid, std_u[:, 1], label="Y std")
        ax.plot(t_mid, std_u[:, 2], label="Z std")
        ax.legend()

    return [Panel("Chunk mean", draw_mean), Panel("Chunk std", draw_std)]


def panels_events(st: State) -> List[Panel]:
    chunk_samples = max(2, int(round(st.set.chunk_sec * st.fs)))
    off_mode = st.set.off_mode.lower()
    if off_mode == "pointwise":
        off_mode = "linear"

    t_mid, mean_v = chunk_means(st.xyz_mm, st.sel.s0, st.sel.s1, chunk_samples, st.fs, off_mode, st.off0, st.off1)
    mean_nt, _ = convert_units_from_volts(mean_v, "nT", st.scale_nt_per_v)

    if st.set.event_use_mag:
        y = np.linalg.norm(mean_nt, axis=1)
    else:
        y = np.max(np.abs(mean_nt), axis=1)

    events_t = detect_events(t_mid, y, threshold=st.set.event_threshold_ntps)

    def draw(ax: plt.Axes):
        ax.set_title(f"Events (threshold={st.set.event_threshold_ntps} nT/s, chunk={st.set.chunk_sec}s)")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("nT (|B|)" if st.set.event_use_mag else "nT (max|axis|)")
        ax.grid(True)
        ax.plot(t_mid, y, label="signal")
        if events_t:
            ax.scatter(events_t, np.interp(events_t, t_mid, y), marker="x", s=60, label="events")
        ax.legend()
        if events_t:
            print(f"[Events] Found {len(events_t)} events. First 10:")
            for te in events_t[:10]:
                print(f"  t={te:.3f}s")

    return [Panel("Events", draw)]


def panels_psd_asd(st: State) -> List[Panel]:
    n = st.sel.s1 - st.sel.s0
    dur_s = n / st.fs
    if dur_s > 3600 and not ask_yesno(f"PSD on {dur_s/3600:.2f} h selection can be slow. Continue?", default="n"):
        raise RuntimeError("PSD cancelled by user. Use a shorter time window.")

    axis = st.set.psd_axis.lower()
    idx = {"x": 0, "y": 1, "z": 2}[axis]

    seg_v = np.asarray(st.xyz_mm[st.sel.s0:st.sel.s1], dtype=np.float32)
    seg_v = apply_offset(seg_v, st.fs, st.sel.s0, st.set.off_mode, st.off0, st.off1, st.sel.point_tmid_s)

    psd_units = st.set.psd_units.lower()
    if psd_units == "v":
        x = seg_v[:, idx].astype(np.float64)
        unit_label = "V"
        asd_pt = False
    elif psd_units == "nt":
        x = (seg_v[:, idx].astype(np.float64) * float(st.scale_nt_per_v[idx]))
        unit_label = "nT"
        asd_pt = False
    else:
        x = (seg_v[:, idx].astype(np.float64) * float(st.scale_nt_per_v[idx]) * 1e-9)
        unit_label = "T"
        asd_pt = st.set.asd_in_pT

    if st.set.psd_detrend.lower() == "linear":
        x = detrend_linear(x)

    f, pxx = welch_psd(
        x,
        fs=st.fs,
        nperseg=int(st.set.welch_nperseg),
        noverlap=int(st.set.welch_noverlap),
        window=st.set.welch_window
    )

    chk = parseval_psd_check(x, f, pxx)
    print(f"[Parseval] var_time = {chk['var_time']:.6g} {unit_label}^2")
    print(f"[Parseval] ∫PSD df  = {chk['var_psd']:.6g} {unit_label}^2")
    print(f"[Parseval] ratio(psd/time) = {chk['ratio']:.4f}  (err={chk['err_pct']:+.2f}%)")

    rms_band = band_rms_from_psd(f, pxx, st.set.band_lo, st.set.band_hi)
    print(f"[PSD] Axis {axis.upper()}, units={unit_label}")
    print(f"[PSD] RMS band {st.set.band_lo}..{st.set.band_hi} Hz ≈ {rms_band:.6g} {unit_label}")

    if st.set.band_lo <= 1.0 <= st.set.band_hi:
        m = (f >= st.set.band_lo) & (f <= st.set.band_hi)
        if np.any(m):
            asd_band = float(np.sqrt(np.mean(pxx[m])))
            if unit_label == "T" and asd_pt:
                print(f"[PSD] ASD band avg ≈ {asd_band*1e12:.3g} pT/√Hz")
            else:
                print(f"[PSD] ASD band avg ≈ {asd_band:.3g} {unit_label}/√Hz")

    def draw_psd(ax: plt.Axes):
        ax.set_title(f"PSD ({axis.upper()}, detrend={st.set.psd_detrend})")
        ax.set_xlabel("f (Hz)")

        pxx_plot = pxx
        psd_ylabel = f"{unit_label}²/Hz"

        if unit_label == "T":
            pxx_plot = pxx * 1e24
            psd_ylabel = "pT²/Hz"

        ax.set_ylabel(psd_ylabel)
        ax.grid(True, which="both", ls=":")
        ax.loglog(f[1:], pxx_plot[1:], label="PSD")  
        ax.legend()

        ax.text(
            0.02, 0.02,
            f"Parseval ratio={chk['ratio']:.3f}  ({chk['err_pct']:+.1f}%)",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7)
        )

    def draw_asd(ax: plt.Axes):
        asd = np.sqrt(pxx)
        ax.set_title(f"ASD ({axis.upper()})")
        ax.set_xlabel("f (Hz)")
        if unit_label == "T" and asd_pt:
            ax.set_ylabel("pT/√Hz")
            ax.loglog(f[1:], asd[1:] * 1e12, label="ASD")
        else:
            ax.set_ylabel(f"{unit_label}/√Hz")
            ax.loglog(f[1:], asd[1:], label="ASD")
        ax.grid(True, which="both", ls=":")
        ax.legend()

    return [Panel("PSD", draw_psd), Panel("ASD", draw_asd)]


def panels_spectrogram(st: State) -> List[Panel]:
    n = st.sel.s1 - st.sel.s0
    dur_s = n / st.fs
    if dur_s > 1800 and not ask_yesno(f"Spectrogram on {dur_s/60:.1f} min selection can be slow. Continue?", default="n"):
        raise RuntimeError("Spectrogram cancelled by user. Use a shorter time window.")

    axis = st.set.spec_axis.lower()
    idx = {"x": 0, "y": 1, "z": 2}[axis]

    seg_v = np.asarray(st.xyz_mm[st.sel.s0:st.sel.s1], dtype=np.float32)
    seg_v = apply_offset(seg_v, st.fs, st.sel.s0, st.set.off_mode, st.off0, st.off1, st.sel.point_tmid_s)

    if st.set.spec_units.lower() == "v":
        x = seg_v[:, idx].astype(np.float64)
        ulabel = "V"
    else:
        x = seg_v[:, idx].astype(np.float64) * float(st.scale_nt_per_v[idx])
        ulabel = "nT"

    if st.set.psd_detrend.lower() == "linear":
        x = detrend_linear(x)

    NFFT = int(st.set.spec_nfft)
    NOV = int(st.set.spec_overlap)
    fmax = float(st.set.spec_fmax)

    def draw(ax: plt.Axes):
        ax.set_title(f"Spectrogram ({axis.upper()}, units={ulabel})")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("f (Hz)")
        Pxx, freqs, bins, im = ax.specgram(
            x,
            NFFT=NFFT,
            Fs=st.fs,
            noverlap=NOV,
            detrend="mean",
            scale="dB",
            cmap="viridis",
        )
        if fmax > 0:
            ax.set_ylim(0, min(fmax, st.fs / 2))
        ax._spec_im = im 

    return [Panel("Spectrogram", draw)]


def panels_allan(st: State) -> List[Panel]:
    chunk_samples = max(2, int(round(st.set.chunk_sec * st.fs)))
    off_mode = st.set.off_mode.lower()
    if off_mode == "pointwise":
        off_mode = "linear"

    t_mid, mean_v = chunk_means(st.xyz_mm, st.sel.s0, st.sel.s1, chunk_samples, st.fs, off_mode, st.off0, st.off1)
    mean_nt, _ = convert_units_from_volts(mean_v, "nT", st.scale_nt_per_v)

    sig = st.set.allan_sig.lower()
    if sig == "x":
        y = mean_nt[:, 0]
        ulabel = "nT"
    elif sig == "y":
        y = mean_nt[:, 1]
        ulabel = "nT"
    elif sig == "z":
        y = mean_nt[:, 2]
        ulabel = "nT"
    else:
        y = np.linalg.norm(mean_nt, axis=1)
        ulabel = "nT"

    dt = float(np.median(np.diff(t_mid))) if len(t_mid) > 1 else float(st.set.chunk_sec)
    dt = dt if (np.isfinite(dt) and dt > 0) else float(st.set.chunk_sec)

    tau_min = max(dt, float(st.set.allan_tau_min))
    tau_max = max(tau_min * 2, float(st.set.allan_tau_max))
    n_tau = int(st.set.allan_tau_n)
    taus = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)

    tau_u, adev = allan_overlapping(y, dt, taus)

    def draw(ax: plt.Axes):
        ax.set_title(f"Allan deviation ({sig}, chunk={st.set.chunk_sec}s)")
        ax.set_xlabel("τ (s)")
        ax.set_ylabel(ulabel)
        ax.grid(True, which="both", ls=":")
        ax.loglog(tau_u, adev)

    return [Panel("Allan", draw)]

def panels_raw_cones(st: State) -> List[Panel]:
    """
    Uses summary_raw.csv only (no offsets) and generates a Plotly cone plot.
    """
    msg = ""

    try:
        x, y, z, bx, by, bz, bmag, src = load_raw_map_from_summary(st.session_dir, st.scale_nt_per_v)
        cmin, cmax = get_cone_cscale_limits(st, bmag)

        out_html, png_note = plot_cones_raw_plotly(
            st.session_dir,
            x, y, z, bx, by, bz, bmag,
            title=f"Raw field map (no offset) - {src}",
            cmin=cmin,
            cmax=cmax,
            sizeref=st.set.cone_sizeref
        )

        msg = (
            f"Opened Plotly cone map.\n"
            f"Color scale: {cmin:.3g} .. {cmax:.3g} nT\n"
            f"Saved: {os.path.basename(out_html)}{png_note}"
        )
    except Exception as e:
        msg = f"Raw cone export failed:\n{e}"

    def draw(ax: plt.Axes):
        ax.axis("off")
        ax.text(0.02, 0.95, "3D cone map (raw, no offset)", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")
        ax.text(0.02, 0.82, msg, transform=ax.transAxes, fontsize=10, va="top")

    return [Panel("Raw cones", draw)]

def panels_aux_time_trace(st: State) -> List[Panel]:
    if st.aux_mm is None or st.aux_n_total <= 0:
        raise RuntimeError("No stream_aux_f32.bin found in this session.")

    axis = st.set.aux_axis_plot.lower()
    t, seg_v = get_segment_aux_for_timeplot(st)

    panels = []

    if axis == "all":
        for k in range(3):
            def make_draw(idx):
                def draw(ax: plt.Axes):
                    ax.set_title(f"AUX{idx+1} time trace (raw volts)")
                    ax.set_xlabel("t (s)")
                    ax.set_ylabel("V")
                    ax.grid(True)
                    ax.plot(t, seg_v[:, idx], label=f"AUX{idx+1}")
                    ax.legend()
                return draw
            panels.append(Panel(f"AUX{k+1} time trace", make_draw(k)))
    else:
        idx = {"1": 0, "2": 1, "3": 2}[axis]

        def draw(ax: plt.Axes):
            ax.set_title(f"AUX{idx+1} time trace (raw volts)")
            ax.set_xlabel("t (s)")
            ax.set_ylabel("V")
            ax.grid(True)
            ax.plot(t, seg_v[:, idx], label=f"AUX{idx+1}")
            ax.legend()

        panels.append(Panel(f"AUX{idx+1} time trace", draw))

    return panels


def panels_aux_fft(st: State) -> List[Panel]:
    if st.aux_mm is None or st.aux_n_total <= 0:
        raise RuntimeError("No stream_aux_f32.bin found in this session.")

    s0, s1 = st.sel.s0, st.sel.s1
    s0 = max(0, min(st.aux_n_total, s0))
    s1 = max(0, min(st.aux_n_total, s1))
    if s1 <= s0:
        raise RuntimeError("Empty AUX selection for FFT.")

    seg_v = np.asarray(st.aux_mm[s0:s1], dtype=np.float32)

    win = st.set.aux_fft_window.lower()
    detr = st.set.aux_fft_detrend.lower()
    axis = st.set.aux_axis_fft.lower()

    def calc_fft(idx):
        x = seg_v[:, idx].astype(np.float64)
        f, amp = single_sided_fft_amplitude(
            x,
            st.fs,
            window=("hann" if win == "hann" else "hamming" if win == "hamming" else "rect"),
            detrend=detr
        )
        peak_i = int(np.argmax(amp[1:])) + 1 if len(amp) > 1 else 0
        print(f"[AUX FFT] Channel AUX{idx+1}, peak ≈ {amp[peak_i]:.6g} V at {f[peak_i]:.3f} Hz")
        return f, amp

    panels = []

    if axis == "all":
        for k in range(3):
            f, amp = calc_fft(k)

            def make_draw(idx, f_local, amp_local):
                def draw(ax: plt.Axes):
                    ax.set_title(f"AUX{idx+1} FFT amplitude spectrum")
                    ax.set_xlabel("f (Hz)")
                    ax.set_ylabel("Amplitude (V)")
                    ax.grid(True, which="both", ls=":")
                    ax.plot(f_local, amp_local)
                    if st.set.aux_fft_fmax > 0:
                        ax.set_xlim(0, min(st.set.aux_fft_fmax, st.fs / 2))
                return draw

            panels.append(Panel(f"AUX{k+1} FFT", make_draw(k, f, amp)))
    else:
        idx = {"1": 0, "2": 1, "3": 2}[axis]
        f, amp = calc_fft(idx)

        def draw(ax: plt.Axes):
            ax.set_title(f"AUX{idx+1} FFT amplitude spectrum")
            ax.set_xlabel("f (Hz)")
            ax.set_ylabel("Amplitude (V)")
            ax.grid(True, which="both", ls=":")
            ax.plot(f, amp)
            if st.set.aux_fft_fmax > 0:
                ax.set_xlim(0, min(st.set.aux_fft_fmax, st.fs / 2))

        panels.append(Panel(f"AUX{idx+1} FFT", draw))

    return panels

def panels_aux_xy_product(st: State) -> List[Panel]:
    """
    AUX XY / hysteresis QA:
      1) XY plot of one AUX channel against another, restricted to the first
         few periods after detected signal onset
      2) text report including product statistics X*Y
    """
    if st.aux_mm is None or st.aux_n_total <= 0:
        raise RuntimeError("No stream_aux_f32.bin found in this session.")

    s0, s1 = st.sel.s0, st.sel.s1
    s0 = max(0, min(st.aux_n_total, s0))
    s1 = max(0, min(st.aux_n_total, s1))
    if s1 <= s0:
        raise RuntimeError("Empty AUX selection.")

    ch_x = st.set.aux_xy_x
    ch_y = st.set.aux_xy_y
    if ch_x not in ("1", "2", "3") or ch_y not in ("1", "2", "3"):
        raise RuntimeError("AUX XY channels must be one of 1, 2, 3.")

    ix = {"1": 0, "2": 1, "3": 2}[ch_x]
    iy = {"1": 0, "2": 1, "3": 2}[ch_y]

    seg_v = np.asarray(st.aux_mm[s0:s1], dtype=np.float32)
    x = seg_v[:, ix].astype(np.float64)
    y = seg_v[:, iy].astype(np.float64)

    trig_sig = x if st.set.aux_xy_trigger_channel.lower() == "x" else y

    onset = find_signal_onset_by_envelope(
        trig_sig,
        sigma_thresh=float(st.set.aux_xy_trigger_sigma),
        smooth=max(20, int(round(0.01 * st.fs)))
    )

    period_sig = x if st.set.aux_xy_period_source.lower() == "x" else y

    period_samples = estimate_period_samples_from_fft(period_sig[onset:], st.fs)

    if period_samples is not None and period_samples > 1:
        n_periods = max(1, int(st.set.aux_xy_periods_to_show))
        n_keep = min(len(x) - onset, n_periods * period_samples)

        a = onset
        b = onset + n_keep

        x_view = x[a:b]
        y_view = y[a:b]
    else:
        a = onset
        b = min(len(x), onset + st.set.aux_xy_max_points)
        x_view = x[a:b]
        y_view = y[a:b]

    n_view = len(x_view)
    _, step = downsample_slice(n_view, st.set.aux_xy_max_points)
    x_plot = x_view[::step]
    y_plot = y_view[::step]

    prod = x_view * y_view

    stats_lines = [
        "AUX XY / hysteresis QA",
        "",
        f"Selection samples : {s0} .. {s1}",
        f"Selection duration: {(s1 - s0) / st.fs:.6g} s",
        f"X channel         : AUX{ch_x}",
        f"Y channel         : AUX{ch_y}",
        f"Trigger channel   : AUX{ch_x if st.set.aux_xy_trigger_channel.lower() == 'x' else ch_y}",
        f"Trigger sigma     : {st.set.aux_xy_trigger_sigma:.3g}",
        f"Onset sample      : {onset}",
        f"Onset time        : {(s0 + onset) / st.fs:.6g} s",
        f"Period source     : AUX{ch_x if st.set.aux_xy_period_source.lower() == 'x' else ch_y}",
        f"Estimated period  : {period_samples} samples" if period_samples is not None else "Estimated period  : failed",
        f"Periods shown     : {st.set.aux_xy_periods_to_show}" if period_samples is not None else "Periods shown     : fallback window",
        f"Plot points       : {len(x_plot)} (step={step})",
        "",
        f"mean(X*Y)         = {np.mean(prod):.6g}",
        f"std(X*Y)          = {np.std(prod, ddof=1):.6g}" if len(prod) > 1 else f"std(X*Y)          = 0",
        f"min(X*Y)          = {np.min(prod):.6g}",
        f"max(X*Y)          = {np.max(prod):.6g}",
        f"rms(X*Y)          = {np.sqrt(np.mean(prod**2)):.6g}",
    ]
    text = "\n".join(stats_lines)

    def draw_xy(ax: plt.Axes):
        ax.set_title(
            f"AUX hysteresis / XY plot: AUX{ch_y} vs AUX{ch_x}",
            fontfamily="STIX Two Text",
            fontsize=13
        )
        ax.set_xlabel(
            f"AUX{ch_x} (V)",
            fontfamily="STIX Two Text",
            fontsize=12
        )
        ax.set_ylabel(
            f"AUX{ch_y} (V)",
            fontfamily="STIX Two Text",
            fontsize=12
        )
        ax.grid(True)

        if st.set.aux_xy_plot_mode.lower() == "scatter":
            ax.scatter(x_plot, y_plot, s=4)
        else:
            ax.plot(x_plot, y_plot, linewidth=0.8)

    def draw_text(ax: plt.Axes):
        ax.axis("off")
        ax.text(
            0.02, 0.98, text,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            family="serif"
        )

    return [
        Panel("AUX XY plot", draw_xy),
        Panel("AUX XY product stats", draw_text),
    ]


def panels_fieldmap_qa(st: State) -> List[Panel]:
    """
    Fieldmap QA from corrected summary.csv.
    Requires:
      - complete corrected fieldmap
      - START and END offsets present
    """
    msg_lines = []

    try:
        fm = load_corrected_fieldmap_from_summary(st.session_dir, st.cfg)
        m = compute_fieldmap_metrics(fm)

        off0 = m["off0_nt"]
        off1 = m["off1_nt"]
        doff = m["doff_nt"]

        msg_lines = [
            "Fieldmap QA metrics (corrected summary.csv):",
            "",
            f"B_c            = {m['B_center_nT']:.6g} nT  at {m['B_center_point']}",
            f"B_max          = {m['B_max_nT']:.6g} nT  at {m['B_max_point']}",
            f"B_min          = {m['B_min_nT']:.6g} nT  at {m['B_min_point']}",
            f"ΔB             = {m['dB_nT']:.6g} nT",
            "",
            f"B_mean,V       = {m['B_mean_V_nT']:.6g} nT",
            f"σ_B,V          = {m['B_std_V_nT']:.6g} nT",
            "",
            f"ΔBx_adj,max    = {m['dBx_adj_max_nT']:.6g} nT",
            f"ΔBy_adj,max    = {m['dBy_adj_max_nT']:.6g} nT",
            f"ΔBz_adj,max    = {m['dBz_adj_max_nT']:.6g} nT",
            "",
            f"Gx,max         = {m['Gx_max_nT_per_m']:.6g} nT/m",
            f"Gy,max         = {m['Gy_max_nT_per_m']:.6g} nT/m",
            f"Gz,max         = {m['Gz_max_nT_per_m']:.6g} nT/m",
            "",
            f"ΔB_RMS,adj     = {m['dB_RMS_adj_nT']:.6g} nT",
            f"G_RMS          = {m['G_RMS_nT_per_m']:.6g} nT/m",
            "",
            f"N_gx           = {m['N_gx']}",
            f"N_gy           = {m['N_gy']}",
            f"N_gz           = {m['N_gz']}",
            f"N_g            = {m['N_g']}",
            f"Grid spacing   = dx={m['dx_m']:.6g} m, dy={m['dy_m']:.6g} m, dz={m['dz_m']:.6g} m",
            "",
            f"START offset   = X={off0[0]:.6g} nT, Y={off0[1]:.6g} nT, Z={off0[2]:.6g} nT",
            f"END offset     = X={off1[0]:.6g} nT, Y={off1[1]:.6g} nT, Z={off1[2]:.6g} nT",
            f"Offset drift   = X={doff[0]:+.6g} nT, Y={doff[1]:+.6g} nT, Z={doff[2]:+.6g} nT",
        ]

        print("\n[Fieldmap QA]")
        for line in msg_lines:
            print(line)

    except Exception as e:
        msg_lines = [
            "Fieldmap QA unavailable.",
            "",
            str(e),
        ]
        print(f"[Fieldmap QA] ERROR: {e}")

    text = "\n".join(msg_lines)

    def draw(ax: plt.Axes):
        ax.axis("off")
        ax.text(
            0.02, 0.98, text,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            family="monospace"
        )

    return [Panel("Fieldmap QA", draw)]

def panels_fieldmap_replot(st: State) -> List[Panel]:
    """
    Replot corrected 3D fieldmap from summary.csv using the same Plotly cone style
    as the GUI/raw cone export.
    """
    msg = ""

    try:
        fm = load_corrected_fieldmap_from_summary(st.session_dir, st.cfg)

        cmin, cmax = get_cone_cscale_limits(st, fm["bmag_nt"])

        out_html, png_note = plot_cones_corrected_plotly(
            st.session_dir,
            fm["x"], fm["y"], fm["z"],
            fm["bx_corr_nt"], fm["by_corr_nt"], fm["bz_corr_nt"], fm["bmag_nt"],
            title="Corrected field map (cone replot)",
            cmin=cmin,
            cmax=cmax,
            basename="cones_corr_replot"
        )

        msg = (
            f"Opened corrected Plotly cone map.\n"
            f"Color scale: {cmin:.3g} .. {cmax:.3g} nT\n"
            f"Saved: {os.path.basename(out_html)}{png_note}"
        )
    except Exception as e:
        msg = f"Corrected cone replot failed:\n{e}"

    def draw(ax: plt.Axes):
        ax.axis("off")
        ax.text(
            0.02, 0.95,
            "Corrected 3D fieldmap replot",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top"
        )
        ax.text(
            0.02, 0.82,
            msg,
            transform=ax.transAxes,
            fontsize=10,
            va="top"
        )

    return [Panel("Fieldmap replot", draw)]

def panels_fieldmap_difference_replot(st: State) -> List[Panel]:
    """
    Load a second complete corrected fieldmap, subtract it from the current one,
    and replot the residual vector field with the corrected cone plot style.
    """
    msg = ""

    try:
        fm_a = load_corrected_fieldmap_from_summary(st.session_dir, st.cfg)

        folder_b = pick_second_session_folder("Select second corrected fieldmap folder")
        if not folder_b:
            raise RuntimeError("No second folder selected.")

        cfg_b_path = os.path.join(folder_b, "config.json")
        if not os.path.exists(cfg_b_path):
            raise RuntimeError(f"Missing config.json in second folder: {folder_b}")
        cfg_b = load_json(cfg_b_path)
        fm_b = load_corrected_fieldmap_from_summary(folder_b, cfg_b)

        fm_mean, fm_diff = compute_fieldmap_mean_and_diff(fm_a, fm_b)

        m_mean = compute_fieldmap_metrics(fm_mean)
        m_diff = compute_fieldmap_metrics(fm_diff)

        cmin_m, cmax_m = get_cone_cscale_limits(st, fm_mean["bmag_nt"])

        out_html_mean, png_note_mean = plot_cones_corrected_plotly(
            st.session_dir,
            fm_mean["x"], fm_mean["y"], fm_mean["z"],
            fm_mean["bx_corr_nt"], fm_mean["by_corr_nt"], fm_mean["bz_corr_nt"], fm_mean["bmag_nt"],
            title="Mean field map (common component)",
            cmin=cmin_m,
            cmax=cmax_m,
            basename="cones_mean"
        )

        cmin_d, cmax_d = get_cone_cscale_limits(st, fm_diff["bmag_nt"])

        out_html_diff, png_note_diff = plot_cones_corrected_plotly(
            st.session_dir,
            fm_diff["x"], fm_diff["y"], fm_diff["z"],
            fm_diff["bx_corr_nt"], fm_diff["by_corr_nt"], fm_diff["bz_corr_nt"], fm_diff["bmag_nt"],
            title="Half-difference field map (sign-changing component)",
            cmin=cmin_d,
            cmax=cmax_d,
            basename="cones_diff"
        )

        msg = (
            f"Generated mean and half-difference fieldmaps.\n"
            f"Second folder: {folder_b}\n\n"

            f"MEAN MAP\n"
            f"  B_c        = {m_mean['B_center_nT']:.6g} nT\n"
            f"  B_mean,V   = {m_mean['B_mean_V_nT']:.6g} nT\n"
            f"  σ_B,V      = {m_mean['B_std_V_nT']:.6g} nT\n"
            f"  B_max      = {m_mean['B_max_nT']:.6g} nT\n"
            f"  B_min      = {m_mean['B_min_nT']:.6g} nT\n"
            f"  ΔB         = {m_mean['dB_nT']:.6g} nT\n"
            f"  ΔB_RMS,adj = {m_mean['dB_RMS_adj_nT']:.6g} nT\n"
            f"  G_RMS      = {m_mean['G_RMS_nT_per_m']:.6g} nT/m\n"
            f"  Scale      = {cmin_m:.3g} .. {cmax_m:.3g} nT\n"
            f"  Saved      = {os.path.basename(out_html_mean)}{png_note_mean}\n\n"

            f"HALF-DIFFERENCE MAP\n"
            f"  B_c        = {m_diff['B_center_nT']:.6g} nT\n"
            f"  B_mean,V   = {m_diff['B_mean_V_nT']:.6g} nT\n"
            f"  σ_B,V      = {m_diff['B_std_V_nT']:.6g} nT\n"
            f"  B_max      = {m_diff['B_max_nT']:.6g} nT\n"
            f"  B_min      = {m_diff['B_min_nT']:.6g} nT\n"
            f"  ΔB         = {m_diff['dB_nT']:.6g} nT\n"
            f"  ΔB_RMS,adj = {m_diff['dB_RMS_adj_nT']:.6g} nT\n"
            f"  G_RMS      = {m_diff['G_RMS_nT_per_m']:.6g} nT/m\n"
            f"  Scale      = {cmin_d:.3g} .. {cmax_d:.3g} nT\n"
            f"  Saved      = {os.path.basename(out_html_diff)}{png_note_diff}"
        )

    except Exception as e:
        msg = f"Residual fieldmap replot failed:\n{e}"

    def draw(ax: plt.Axes):
        ax.axis("off")
        ax.text(
            0.02, 0.95,
            "Residual 3D fieldmap replot",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top"
        )
        ax.text(
            0.02, 0.82,
            msg,
            transform=ax.transAxes,
            fontsize=10,
            va="top"
        )

    return [Panel("Fieldmap difference", draw)]

def panels_aux_points_qa(st: State) -> List[Panel]:
    """
    AUX points QA:
      1) line plot of corrected |B| over AUX point index
      2) text report of consecutive point-to-point shifts
    """
    rows = load_aux_points(st.session_dir)
    if not rows:
        raise RuntimeError("No aux_points.csv found in this session.")

    idx = []
    comments = []
    bx = []
    by = []
    bz = []
    bmag = []

    for i, r in enumerate(rows, start=1):
        try:
            bx_i = float(r["Bx_corr_nT"])
            by_i = float(r["By_corr_nT"])
            bz_i = float(r["Bz_corr_nT"])
            bm_i = float(r["Bmag_corr_nT"])
        except Exception:
            raise RuntimeError("aux_points.csv is missing corrected AUX point columns.")

        idx.append(i)
        comments.append(r.get("comment", "").strip())
        bx.append(bx_i)
        by.append(by_i)
        bz.append(bz_i)
        bmag.append(bm_i)

    idx = np.asarray(idx, dtype=int)
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    bz = np.asarray(bz, dtype=float)
    bmag = np.asarray(bmag, dtype=float)

    lines = []
    lines.append("AUX points QA")
    lines.append("")
    lines.append(f"Number of AUX points: {len(idx)}")
    lines.append("")

    for i in range(len(idx)):
        cmt = comments[i] if comments[i] else "(blank)"
        lines.append(
            f"{idx[i]:>2d}: "
            f"Bx={bx[i]: .6g} nT, "
            f"By={by[i]: .6g} nT, "
            f"Bz={bz[i]: .6g} nT, "
            f"|B|={bmag[i]: .6g} nT, "
            f"comment={cmt}"
        )

    if len(idx) >= 2:
        lines.append("")
        lines.append("Consecutive shifts:")
        lines.append("")

        for i in range(len(idx) - 1):
            dbx = bx[i + 1] - bx[i]
            dby = by[i + 1] - by[i]
            dbz = bz[i + 1] - bz[i]
            dbm = bmag[i + 1] - bmag[i]

            c0 = comments[i] if comments[i] else "(blank)"
            c1 = comments[i + 1] if comments[i + 1] else "(blank)"

            lines.append(
                f"{idx[i]:>2d}->{idx[i+1]:>2d}: "
                f"ΔBx={dbx:+.6g} nT, "
                f"ΔBy={dby:+.6g} nT, "
                f"ΔBz={dbz:+.6g} nT, "
                f"Δ|B|={dbm:+.6g} nT   "
                f"[{c0} -> {c1}]"
            )

    text = "\n".join(lines)

    def draw_path(ax: plt.Axes):
        ax.set_title(r"AUX points: corrected $|B|$")
        ax.set_xlabel("AUX point index")
        ax.set_ylabel(r"$|\overline{B}|$ (nT)")
        ax.grid(True)

        ax.plot(idx, bmag, marker="o", linewidth=1.5)

        for i_pt, b_pt in zip(idx, bmag):
            ax.text(i_pt, b_pt, f"{i_pt}", fontsize=8, ha="left", va="bottom")

    def draw_text(ax: plt.Axes):
        ax.axis("off")
        ax.text(
            0.02, 0.98, text,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            family="serif"
        )

    return [
        Panel("AUX points magnitude path", draw_path),
        Panel("AUX points shift report", draw_text),
    ]

# ============================================================
# Multi-analysis runner with subplots (Option A)
# ============================================================

ANALYSES = {
    "1": ("Time trace", panels_time_trace),
    "2": ("Chunk mean", panels_chunk_stats),
    "3": ("PSD + ASD", panels_psd_asd),
    "4": ("Spectrogram", panels_spectrogram),
    "5": ("Event detection", panels_events),
    "6": ("Allan deviation", panels_allan),
    "7": ("3D cone map (raw from summary_raw.csv)", panels_raw_cones),
    "8": ("AUX time trace (raw V)", panels_aux_time_trace),
    "9": ("AUX FFT amplitude spectrum", panels_aux_fft),
    "10": ("Fieldmap QA (corrected summary.csv)", panels_fieldmap_qa),
    "11": ("Replot corrected 3D fieldmap (thesis style)", panels_fieldmap_replot),
    "12": ("Residual 3D fieldmap (map 1 - map 2)", panels_fieldmap_difference_replot),
    "13": ("AUX points QA", panels_aux_points_qa),
        "14": ("AUX XY / hysteresis QA", panels_aux_xy_product),
}



def run_analyses(st: State, keys: List[str]):
    panels: List[Panel] = []
    needs_colorbar_axes = []

    for k in keys:
        name, builder = ANALYSES[k]
        ps = builder(st)
        panels.extend(ps)

    n = len(panels)
    if n == 0:
        print("[WARN] No panels to plot.")
        return

    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig_w = 7.0 * ncols
    fig_h = 4.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for i, p in enumerate(panels):
        ax = axes[i]
        p.draw(ax)
        if hasattr(ax, "_spec_im"):
            needs_colorbar_axes.append(ax)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    for ax in needs_colorbar_axes:
        im = getattr(ax, "_spec_im", None)
        if im is not None:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB")

    fig.suptitle(f"{st.sel.label} | units={st.set.units} | offset={st.set.off_mode}", fontsize=12)
    plt.show()


# ============================================================
# Main menu loop
# ============================================================

def main():
    print("\n=== FieldMap QA + Analysis Tool (streamlined menu) ===\n")

    folder = pick_folder_dialog()
    if folder is None:
        folder = ask("Paste session folder path", default=os.getcwd())
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Not a folder: {folder}")

    cfg_path = os.path.join(folder, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config.json in: {folder}")

    cfg = load_json(cfg_path)
    scale_nt_per_v = load_scale_nt_per_v(cfg)
    xyz_mm, n_total = memmap_stream_xyz(folder)
    aux_mm, aux_n_total = memmap_stream_aux(folder)
    fs = infer_fs(folder, cfg)
    off0, off1 = load_offsets(folder)
    summary_rows, summary_name = load_summary(folder)

    st = State(
        session_dir=folder,
        cfg=cfg,
        fs=fs,
        scale_nt_per_v=scale_nt_per_v,
        xyz_mm=xyz_mm,
        n_total=n_total,
        aux_mm=aux_mm,
        aux_n_total=aux_n_total,
        off0=off0,
        off1=off1,
        summary_rows=summary_rows,
        summary_name=summary_name,
    )

    choose_selection(st)
    choose_processing(st)

    while True:
        print_header(st)
        print("Menu:")
        print("  1) Run ONE analysis")
        print("  2) Run MULTIPLE analyses (comma-separated)")
        print("  3) Change selection (point / time window)")
        print("  4) Change processing (units / offset)")
        print("  5) Presets")
        print("  6) Edit analysis defaults (chunk/PSD/spectrogram/events/allan)")
        print("  0) Quit")

        cmd = ask("Select", default="1", valid=["0", "1", "2", "3", "4", "5", "6"])
        if cmd == "0":
            return

        if cmd == "3":
            choose_selection(st)
            continue

        if cmd == "4":
            choose_processing(st)
            continue

        if cmd == "5":
            try:
                bundle = apply_preset(st)
                run_analyses(st, bundle)
            except Exception as e:
                print(f"[ERROR] {e}")
            continue

        if cmd == "6":
            st.set.axis_plot = ask("Time-trace axis (x/y/z/xyz)", default=st.set.axis_plot, valid=["x", "y", "z", "xyz"])
            st.set.show_mag = ask_yesno("Time-trace show |B|?", default="y" if st.set.show_mag else "n")
            st.set.max_plot_points = ask_int("Max plot points", default=st.set.max_plot_points, min_val=1000)
            st.set.show_point_windows_in_trace = ask("Overlay recorded point windows in time trace (y/n)", default="y" if st.set.show_point_windows_in_trace else "n", valid=["y", "n"]) == "y"

            st.set.chunk_sec = ask_float("Chunk length (s)", default=st.set.chunk_sec, min_val=0.01)

            st.set.psd_axis = ask("PSD axis (x/y/z)", default=st.set.psd_axis, valid=["x", "y", "z"])
            st.set.psd_units = ask("PSD units (V/nT/T)", default=st.set.psd_units, valid=["V", "nT", "T"])
            st.set.psd_detrend = ask("PSD detrend (none/linear)", default=st.set.psd_detrend, valid=["none", "linear"])
            st.set.welch_nperseg = ask_int("Welch nperseg", default=st.set.welch_nperseg, min_val=128)
            st.set.welch_noverlap = ask_int("Welch noverlap", default=st.set.welch_noverlap, min_val=0, max_val=st.set.welch_nperseg-1)
            st.set.welch_window = ask("Welch window (hann/hamming)", default=st.set.welch_window, valid=["hann", "hamming"])
            st.set.band_lo = ask_float("RMS band f_lo (Hz)", default=st.set.band_lo, min_val=0.0)
            st.set.band_hi = ask_float("RMS band f_hi (Hz)", default=st.set.band_hi, min_val=0.0)
            st.set.asd_in_pT = ask_yesno("If PSD units=T, show ASD as pT/√Hz?", default="y" if st.set.asd_in_pT else "n")

            st.set.spec_axis = ask("Spectrogram axis (x/y/z)", default=st.set.spec_axis, valid=["x", "y", "z"])
            st.set.spec_units = ask("Spectrogram units (V/nT)", default=st.set.spec_units, valid=["V", "nT"])
            st.set.spec_nfft = ask_int("Spectrogram NFFT", default=st.set.spec_nfft, min_val=128)
            st.set.spec_overlap = ask_int("Spectrogram overlap", default=st.set.spec_overlap, min_val=0, max_val=st.set.spec_nfft-1)
            st.set.spec_fmax = ask_float("Spectrogram fmax (Hz, 0=auto)", default=st.set.spec_fmax, min_val=0.0)

            st.set.event_threshold_ntps = ask_float("Event threshold (nT/s)", default=st.set.event_threshold_ntps, min_val=0.0)
            st.set.event_use_mag = ask_yesno("Events on |B| (else max|axis|)", default="y" if st.set.event_use_mag else "n")

            st.set.allan_sig = ask("Allan signal (x/y/z/mag)", default=st.set.allan_sig, valid=["x", "y", "z", "mag"])
            st.set.allan_tau_min = ask_float("Allan tau min (s)", default=st.set.allan_tau_min, min_val=0.0)
            st.set.allan_tau_max = ask_float("Allan tau max (s)", default=st.set.allan_tau_max, min_val=0.0)
            st.set.allan_tau_n = ask_int("Allan tau points", default=st.set.allan_tau_n, min_val=5, max_val=200)

            st.set.aux_axis_plot = ask("AUX time-trace channel (all/1/2/3)", default=st.set.aux_axis_plot, valid=["all", "1", "2", "3"])
            st.set.aux_axis_fft = ask("AUX FFT channel (all/1/2/3)", default=st.set.aux_axis_fft, valid=["all", "1", "2", "3"])
            st.set.aux_fft_window = ask("AUX FFT window (hann/hamming/rect)", default=st.set.aux_fft_window, valid=["hann", "hamming", "rect"])
            st.set.aux_fft_detrend = ask("AUX FFT detrend (mean/linear)", default=st.set.aux_fft_detrend, valid=["mean", "linear"])
            st.set.aux_fft_fmax = ask_float("AUX FFT fmax (Hz, 0=full)", default=st.set.aux_fft_fmax, min_val=0.0)
            st.set.aux_xy_x = ask("AUX XY x-channel (1/2/3)", default=st.set.aux_xy_x, valid=["1", "2", "3"])
            st.set.aux_xy_y = ask("AUX XY y-channel (1/2/3)", default=st.set.aux_xy_y, valid=["1", "2", "3"])
            st.set.aux_xy_period_source = ask("AUX XY period source (x/y)", default=st.set.aux_xy_period_source, valid=["x", "y"])
            st.set.aux_xy_periods_to_show = ask_int("AUX XY periods to show", default=st.set.aux_xy_periods_to_show, min_val=1)
            st.set.aux_xy_plot_mode = ask("AUX XY plot mode (line/scatter)", default=st.set.aux_xy_plot_mode, valid=["line", "scatter"])
            st.set.aux_xy_max_points = ask_int("AUX XY max plot points", default=st.set.aux_xy_max_points, min_val=1000)
            st.set.aux_xy_trigger_channel = ask("AUX XY trigger channel (x/y)", default=st.set.aux_xy_trigger_channel, valid=["x", "y"])
            st.set.aux_xy_trigger_sigma = ask_float("AUX XY trigger sigma", default=st.set.aux_xy_trigger_sigma, min_val=0.5)

            st.set.cone_cmin_nt = ask_float("Cone color scale min (nT)", default=st.set.cone_cmin_nt)
            st.set.cone_cmax_mode = ask("Cone color scale mode (fixed/auto)", default=st.set.cone_cmax_mode, valid=["fixed", "auto"])
            st.set.cone_cmax_fixed_nt = ask_float("Cone color scale max if fixed (nT)", default=st.set.cone_cmax_fixed_nt)

            st.set.cone_size_ref_nt = ask_float("Cone size reference (nT)", default=st.set.cone_size_ref_nt)
            st.set.cone_sizeref = ask_float("Cone sizeref", default=st.set.cone_sizeref)
            continue

        if cmd == "1":
            print("\nAnalyses:")
            for k, (nm, _) in ANALYSES.items():
                print(f"  {k}) {nm}")
            k = ask("Choose analysis", default="1", valid=list(ANALYSES.keys()))
            try:
                run_analyses(st, [k])
            except Exception as e:
                print(f"[ERROR] {e}")
            continue

        if cmd == "2":
            print("\nAnalyses:")
            for k, (nm, _) in ANALYSES.items():
                print(f"  {k}) {nm}")
            s = ask("Enter comma-separated list (e.g. 1,5 or 3,4)", default="1,5")
            keys = [x.strip() for x in s.split(",") if x.strip()]
            keys = [k for k in keys if k in ANALYSES]
            if not keys:
                print("[WARN] No valid analyses selected.")
                continue
            try:
                run_analyses(st, keys)
            except Exception as e:
                print(f"[ERROR] {e}")
            continue


if __name__ == "__main__":
    main()
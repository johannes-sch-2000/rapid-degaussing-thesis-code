import os
import time
import csv
import json
import threading
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, List, Tuple, Dict

import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

import importlib.util
import ctypes.util
import platform
import sys

NI_OK = False
NI_IMPORT_ERROR = None
NI_DIAG = {}

try:
    spec = importlib.util.find_spec("nidaqmx")
    NI_DIAG["nidaqmx_spec_found"] = (spec is not None)

    import nidaqmx
    from nidaqmx.constants import AcquisitionType
    from nidaqmx.stream_readers import AnalogMultiChannelReader

    NI_OK = True
    NI_DIAG["nidaqmx_version"] = getattr(nidaqmx, "__version__", "unknown")

except Exception as e:
    NI_OK = False
    NI_IMPORT_ERROR = repr(e)

try:
    NI_DIAG["python_executable"] = sys.executable
    NI_DIAG["python_version"] = sys.version
    NI_DIAG["architecture"] = platform.architecture()[0]
    NI_DIAG["frozen"] = bool(getattr(sys, "frozen", False))
    NI_DIAG["nicaiu_find_library"] = ctypes.util.find_library("nicaiu")
except Exception as e:
    NI_DIAG["diag_error"] = repr(e)



# =========================
# Defaults / Config
# =========================

@dataclass
class AppConfig:
    out_dir_root: str = r"C:\Users\Johannes\masterthesis-msr-degaussing\data\fieldmaps"

    ch_x: str = "cDAQ1Mod2/ai0"
    ch_y: str = "cDAQ1Mod2/ai1"
    ch_z: str = "cDAQ1Mod2/ai2"

    enable_aux: bool = True
    ch_aux1: str = "cDAQ1Mod3/ai0"
    ch_aux2: str = "cDAQ1Mod3/ai1"
    ch_aux3: str = "cDAQ1Mod3/ai2"

    fs_req: float = 2000.0
    chunk_s: float = 0.10
    min_chunk_samps: int = 200

    plot_window_s: float = 10.0
    plot_decim: int = 20

    flip_capture_s: float = 1.0
    point_capture_s: float = 1.0

    grid_n: int = 5
    grid_spacing_m: float = 0.25

    scale_nt_per_v: np.ndarray = field(default_factory=lambda: np.array([7000.0, 7000.0, 7000.0], dtype=float))


    ai_min: float = -10.0
    ai_max: float = 10.0

    qc_clip_margin: float = 0.98
    qc_std_warn_v: float = 0.005
    qc_require_start_offset: bool = True

    cone_sizemode: str = "scaled"
    cone_sizeref: float = 0.6
    colorscale: str = "Jet"
    ax_range: Tuple[float, float] = (-0.55, 0.55)
    camera_eye: Tuple[float, float, float] = (-1.8, -1.8, 0.9)

    flip_sequence: Tuple[str, str, str, str] = (
    "POS1_DEFAULT",
    "POS2_PLATE_Y180",
    "POS3_SIDE_Z90",
    "POS4_SIDE_X180",
)


def timestamp_tag():
    return time.strftime("%Y%m%d_%H%M%S")


def make_session_dir(root: str) -> str:
    d = os.path.join(root, f"map_{timestamp_tag()}")
    os.makedirs(d, exist_ok=True)
    return d


# =========================
# NI helper: device + channel listing
# =========================

def list_ni_devices() -> List[str]:
    if not NI_OK:
        return []
    try:
        sys = nidaqmx.system.System.local()
        return [dev.name for dev in sys.devices]
    except Exception:
        return []


def list_ai_channels(device_name: str) -> List[str]:
    """
    Returns full physical channel strings like Dev1/ai0, Dev1/ai1 ...
    """
    if not NI_OK:
        return []
    try:
        dev = nidaqmx.system.device.Device(device_name)
        chans = []
        for ch in dev.ai_physical_chans:
            chans.append(ch.name)
        return chans
    except Exception:
        return []


# =========================
# Grid order
# =========================

def scan_points_min_movement(n: int):
    """
    Physical order (x,y,z):
      - x changes fastest (snake within a row)
      - y changes next (rows)
      - z changes slowest (planes)
    Minimal movement:
      - z odd planes: y goes 1..n
      - z even planes: y goes n..1
      - x direction alternates by (y+z) parity so the path is continuous.
    """
    pts = []
    for z in range(1, n + 1):
        y_range = range(1, n + 1) if (z % 2 == 1) else range(n, 0, -1)
        for y in y_range:
            x_range = range(1, n + 1) if ((y + z) % 2 == 0) else range(n, 0, -1)
            for x in x_range:
                pts.append((x, y, z))
    return pts


def idx_to_xyz_m(ix, iy, iz, n, spacing_m):
    x = (ix - (n + 1) / 2) * spacing_m
    y = (iy - (n + 1) / 2) * spacing_m
    z = (iz - (n + 1) / 2) * spacing_m
    return x, y, z


# =========================
# Flip offset math + QC
# =========================

def compute_offset_from_minmove_flips(means_v: Dict[str, np.ndarray]):
    """
    Four-position offset method:
      POS1_DEFAULT      : default orientation, gives first values for x and z
      POS2_PLATE_Y180   : rotated 180° on plate around y, gives second values for x and z
      POS3_SIDE_Z90     : sensor rotated 90° around z onto its side, gives first value for y
      POS4_SIDE_X180    : from POS3, rotated 180° on plate around x, gives second value for y

    Assumptions:
      - POS1 -> POS2 flips x and z, while y is not used for offset
      - POS3 -> POS4 flips y
      - offsets are orientation-independent

    Returns:
      offset_v (3,) in V
      field_v  (3,) in V
      qc dict   (all in V)
    """
    P1 = np.asarray(means_v["POS1_DEFAULT"], dtype=np.float64)
    P2 = np.asarray(means_v["POS2_PLATE_Y180"], dtype=np.float64)
    P3 = np.asarray(means_v["POS3_SIDE_Z90"], dtype=np.float64)
    P4 = np.asarray(means_v["POS4_SIDE_X180"], dtype=np.float64)

    Ox = 0.5 * (P1[0] + P2[0])
    Oz = 0.5 * (P1[2] + P2[2])
    Oy = 0.5 * (P3[1] + P4[1])

    offset_v = np.array([Ox, Oy, Oz], dtype=np.float64)

    Bx = 0.5 * (P1[0] - P2[0])
    Bz = 0.5 * (P1[2] - P2[2])
    By = 0.5 * (P3[1] - P4[1])

    field_v = np.array([Bx, By, Bz], dtype=np.float64)

    qc = {
        "x_flip_mismatch_V": float(abs((P1[0] - Ox) + (P2[0] - Ox))),
        "z_flip_mismatch_V": float(abs((P1[2] - Oz) + (P2[2] - Oz))),
        "y_flip_mismatch_V": float(abs((P3[1] - Oy) + (P4[1] - Oy))),
        "plate_y_invariance_V": float(abs(P1[1] - P2[1])),
        "side_x_invariance_V": float(abs(P3[0] - P4[0])),
        "side_z_invariance_V": float(abs(P3[2] - P4[2])),
    }
    return offset_v, field_v, qc


def qc_check_block(block_v: np.ndarray, cfg: AppConfig):
    """
    block_v: shape (3, N) in Volts
    Returns stats dict and qc fields
    """
    mean = block_v.mean(axis=1)
    std = block_v.std(axis=1)
    vmin = block_v.min(axis=1)
    vmax = block_v.max(axis=1)
    absmax = np.max(np.abs(block_v), axis=1)

    fullscale = max(abs(cfg.ai_min), abs(cfg.ai_max))
    near = cfg.qc_clip_margin * fullscale
    clip_frac = np.mean(np.abs(block_v) >= near, axis=1).astype(np.float32)

    qc_code = 0
    if np.any(clip_frac > 0.0):
        qc_code |= 1
    if np.any(std > cfg.qc_std_warn_v):
        qc_code |= 2

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "min": vmin.astype(np.float32),
        "max": vmax.astype(np.float32),
        "absmax": absmax.astype(np.float32),
        "qc_code": int(qc_code),
        "qc_clip_frac": clip_frac,
    }


# =========================
# Continuous NI acquisition + logging + capture
# =========================

class ContinuousNI:
    """
    Continuous acquisition + ring buffer for monitor + binary stream log.

    stream_raw_f32.bin stores interleaved float32:
      x,y,z,x,y,z,...
    stream_index.csv logs chunk boundaries:
      start_sample, n_samples, wall_time_unix, actual_fs
    """
    def __init__(self, channels_field: List[str], cfg: AppConfig, out_dir: str, aux_channels: Optional[List[str]] = None):
        if not NI_OK:
            raise RuntimeError("nidaqmx not available (NI-DAQmx runtime/driver not installed?)")

        self.field_channels = list(channels_field)               
        self.aux_channels = list(aux_channels or [])            
        self.channels = self.field_channels + self.aux_channels 

        self.field_n = len(self.field_channels)  
        self.aux_n = len(self.aux_channels)     

        self.cfg = cfg
        self.out_dir = out_dir

        self.task = None
        self.reader = None
        self.actual_fs = None

        self.running = False
        self.thread = None

        self.sample_index = 0

        self.ring = deque()  
        self.ring_max_samples = int(cfg.plot_window_s * cfg.fs_req)

        self.bin_path = os.path.join(out_dir, "stream_raw_f32.bin")
        self.idx_path = os.path.join(out_dir, "stream_index.csv")
        self.bin_f = None
        self.idx_f = None
        self.idx_writer = None
        self.aux_bin_path = os.path.join(out_dir, "stream_aux_f32.bin")
        self.aux_f = None

        self.capture_lock = threading.Lock()
        self.capture_mode = None
        self.capture_needed = 0
        self.capture_buf = None
        self.capture_start_index = None
        self.capture_done = threading.Event()

        self.clear_requested = False

    def start(self):
        os.makedirs(self.out_dir, exist_ok=True)

        self.bin_f = open(self.bin_path, "wb")
        self.idx_f = open(self.idx_path, "w", newline="")
        if self.aux_n > 0:
            self.aux_f = open(self.aux_bin_path, "wb")
        self.idx_writer = csv.writer(self.idx_f)
        self.idx_writer.writerow(["start_sample", "n_samples", "wall_time_unix", "actual_fs"])

        self.task = nidaqmx.Task()
        for ch in self.channels:
            self.task.ai_channels.add_ai_voltage_chan(ch, min_val=self.cfg.ai_min, max_val=self.cfg.ai_max)

        self.task.timing.cfg_samp_clk_timing(
            rate=float(self.cfg.fs_req),
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=max(self.cfg.min_chunk_samps, int(self.cfg.fs_req * self.cfg.chunk_s))
        )
        self.actual_fs = float(self.task.timing.samp_clk_rate)
        self.reader = AnalogMultiChannelReader(self.task.in_stream)

        self.running = True
        self.task.start()

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.task is not None:
            try:
                self.task.stop()
            except Exception:
                pass
            self.task.close()
        if self.bin_f:
            self.bin_f.close()
        if self.idx_f:
            self.idx_f.close()
        if self.aux_f:
            self.aux_f.close()

    def request_clear(self):
        self.clear_requested = True

    def request_capture(self, mode: str, duration_s: float) -> bool:
        n = int(round(duration_s * float(self.actual_fs)))
        n = max(n, 2)
        with self.capture_lock:
            if self.capture_mode is not None:
                return False
            self.capture_mode = mode
            self.capture_needed = n
            self.capture_buf = np.zeros((self.field_n, n), dtype=np.float32)
            self.capture_start_index = None
            self.capture_done.clear()
        return True

    def wait_capture(self, timeout=10.0):
        ok = self.capture_done.wait(timeout=timeout)
        if not ok:
            return None
        with self.capture_lock:
            mode = self.capture_mode
            buf = self.capture_buf.copy()
            start_idx = int(self.capture_start_index)
            end_idx = start_idx + buf.shape[1]

            self.capture_mode = None
            self.capture_needed = 0
            self.capture_buf = None
            self.capture_start_index = None

        return {"mode": mode, "data": buf, "start_idx": start_idx, "end_idx": end_idx}

    def _append_ring(self, start_idx: int, data: np.ndarray):
        self.ring.append((start_idx, data))
       
        total = sum(d.shape[1] for _, d in self.ring)
        while total > self.ring_max_samples and len(self.ring) > 1:
            _, d0 = self.ring.popleft()
            total -= d0.shape[1]

    def get_last_window(self, window_s: float):
        n_need = int(round(window_s * float(self.actual_fs)))
        if n_need <= 0 or len(self.ring) == 0:
            return None

        chunks = []
        count = 0
        for start_idx, d in reversed(self.ring):
            chunks.append((start_idx, d))
            count += d.shape[1]
            if count >= n_need:
                break
        chunks.reverse()

        if not chunks:
            return None

        start = int(chunks[0][0])
        data = np.concatenate([d for _, d in chunks], axis=1)

        old_len = data.shape[1]
        if old_len > n_need:
            drop = old_len - n_need
            data = data[:, -n_need:]
            start += drop

        return start, data

    def _loop(self):
        n_samp = max(self.cfg.min_chunk_samps, int(round(float(self.actual_fs) * self.cfg.chunk_s)))
        buf = np.zeros((self.field_n + self.aux_n, n_samp), dtype=np.float64)

        while self.running:
            try:
                wall_t = time.time()
                self.reader.read_many_sample(
                    data=buf,
                    number_of_samples_per_channel=n_samp,
                    timeout=2.0
                )
                block_all = buf.astype(np.float32, copy=False)

                block_field = block_all[:self.field_n, :]               
                np.ascontiguousarray(block_field.T).tofile(self.bin_f)  

                if self.aux_n > 0 and self.aux_f is not None:
                    block_aux = block_all[self.field_n:, :]                 
                    np.ascontiguousarray(block_aux.T).tofile(self.aux_f)    

                self.idx_writer.writerow([self.sample_index, n_samp, wall_t, float(self.actual_fs)])
                self.idx_f.flush()
                self.bin_f.flush()
                if self.aux_f:
                    self.aux_f.flush()

                if self.clear_requested:
                    self.ring.clear()
                    self.clear_requested = False

                self._append_ring(self.sample_index, block_all)

                with self.capture_lock:
                    if self.capture_mode is not None and self.capture_needed > 0:
                        if self.capture_start_index is None:
                            self.capture_start_index = self.sample_index
                        take = min(self.capture_needed, n_samp)
                        filled = self.capture_buf.shape[1] - self.capture_needed
                        self.capture_buf[:, filled:filled + take] = block_field[:, :take]
                        self.capture_needed -= take
                        if self.capture_needed <= 0:
                            self.capture_done.set()

                self.sample_index += n_samp

            except Exception:
                time.sleep(0.05)


# =========================
# Export: CSV + NPZ + map_stats + cones
# =========================

def export_cones(session_dir, cfg: AppConfig, points, mean_nt, force_cmax=None, basename="cones_corr", title="Corrected field map (cones)"):
    if not PLOTLY_OK:
        return

    n = len(points)
    X = np.zeros(n, dtype=float)
    Y = np.zeros(n, dtype=float)
    Z = np.zeros(n, dtype=float)
    for i, (ix, iy, iz) in enumerate(points):
        X[i], Y[i], Z[i] = idx_to_xyz_m(ix, iy, iz, cfg.grid_n, cfg.grid_spacing_m)

    Bx = mean_nt[:, 0].astype(float)
    By = mean_nt[:, 1].astype(float)
    Bz = mean_nt[:, 2].astype(float)
    Bmag = np.sqrt(Bx*Bx + By*By + Bz*Bz)
    cmax = float(force_cmax) if force_cmax is not None else float(np.max(Bmag))

    camera = dict(
        eye=dict(x=cfg.camera_eye[0], y=cfg.camera_eye[1], z=cfg.camera_eye[2]),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
    )
    axis = dict(
        range=list(cfg.ax_range),
        tickmode="array",
        tickvals=[-0.5, -0.25, 0, 0.25, 0.5],
        showbackground=False,
        gridcolor="lightgrey",
        zerolinecolor="grey"
    )

    fig = go.Figure()
    fig.add_trace(go.Cone(
        x=X, y=Y, z=Z,
        u=Bx, v=By, w=Bz,
        colorscale=cfg.colorscale,
        cmin=0, cmax=cmax,
        showscale=True,
        colorbar=dict(title="|B| (nT)", x=1.05, y=0.5, len=0.80, thickness=18),
        sizemode=cfg.cone_sizemode,
        sizeref=cfg.cone_sizeref,
        anchor="tail"
    ))
    fig.add_annotation(
        x=0.02, y=0.02, xref="paper", yref="paper",
        text=f"Max. |B| ≈ {float(np.max(Bmag)):.2f} nT",
        showarrow=False, font=dict(size=16, color="orange")
    )
    fig.update_layout(
        title=title,
        width=900, height=700,
        margin=dict(l=10, r=90, t=70, b=40),
        scene=dict(
            xaxis=dict(axis, title="x (m)"),
            yaxis=dict(axis, title="y (m)"),
            zaxis=dict(axis, title="z (m)"),
            aspectmode="cube",
            camera=camera
        )
    )

    html_path = os.path.join(session_dir, f"{basename}.html")
    png_path  = os.path.join(session_dir, f"{basename}.png")
    fig.write_html(html_path)
    try:
        fig.write_image(png_path, width=1200, height=900, scale=2)
    except Exception:
        pass


def finalize_export(session_dir: str, cfg: AppConfig, ni: ContinuousNI,
                    off0: dict, off1: dict,
                    point_blocks: list, point_stats: list):
    if off0 is None or off1 is None:
        raise RuntimeError("Need START and END offset results before export.")
    if "offset_v" not in off0 or "offset_v" not in off1:
        raise RuntimeError("Offset dict missing offset_v.")

    points = [b["point"] for b in point_blocks]
    n = len(points)
    if n == 0:
        raise RuntimeError("No points recorded.")

    start_idx_arr = np.array([int(b["start_idx"]) for b in point_blocks], dtype=np.int64)
    end_idx_arr   = np.array([int(b["end_idx"])   for b in point_blocks], dtype=np.int64)

    mean_meas = np.zeros((n, 3), dtype=np.float32)
    std_meas  = np.zeros((n, 3), dtype=np.float32)
    pt_min    = np.zeros((n, 3), dtype=np.float32)
    pt_max    = np.zeros((n, 3), dtype=np.float32)
    pt_absmax = np.zeros((n, 3), dtype=np.float32)
    tmid_s    = np.zeros(n, dtype=np.float64)

    qc_code      = np.zeros(n, dtype=np.int16)
    qc_clip_frac = np.zeros((n, 3), dtype=np.float32)

    n_samp = int(point_blocks[0]["data"].shape[1])
    raw_points = np.zeros((n, 3, n_samp), dtype=np.float32)

    for i, (blk, st) in enumerate(zip(point_blocks, point_stats)):
        raw = blk["data"].astype(np.float32, copy=False)
        if raw.shape[1] != n_samp:
            n_s = min(n_samp, raw.shape[1])
            raw_points[i, :, :n_s] = raw[:, :n_s]
        else:
            raw_points[i] = raw

        mean_meas[i] = st["mean"]
        std_meas[i]  = st["std"]
        pt_min[i]    = st["min"]
        pt_max[i]    = st["max"]
        pt_absmax[i] = st["absmax"]

        if ("tmid_s" in st) and np.isfinite(st["tmid_s"]):
            tmid_s[i] = float(st["tmid_s"])
        else:
            tmid_s[i] = 0.5 * (start_idx_arr[i] + end_idx_arr[i]) / float(ni.actual_fs)
            st["tmid_s"] = float(tmid_s[i])

        qc_code[i] = int(st.get("qc_code", 0))
        qc_clip_frac[i, :] = np.asarray(st.get("qc_clip_frac", [0, 0, 0]), dtype=np.float32)

    off0_t   = float(off0.get("tmid_s", tmid_s[0]))
    off1_t   = float(off1.get("tmid_s", off0_t))
    off0_vec = np.asarray(off0["offset_v"], dtype=np.float64).reshape(3)
    off1_vec = np.asarray(off1["offset_v"], dtype=np.float64).reshape(3)

    if off1_t <= off0_t + 1e-12:
        off_at_points = np.repeat(off0_vec[None, :], repeats=n, axis=0)
    else:
        a = (tmid_s - off0_t) / (off1_t - off0_t)
        a = np.clip(a, 0.0, 1.0)
        off_at_points = off0_vec[None, :] + a[:, None] * (off1_vec[None, :] - off0_vec[None, :])

    mean_corr_v = (mean_meas.astype(np.float64) - off_at_points).astype(np.float32)

    scale = np.asarray(cfg.scale_nt_per_v, dtype=np.float64).reshape(3)
    mean_meas_nt = (mean_meas.astype(np.float64) * scale).astype(np.float32)
    mean_corr_nt = (mean_corr_v.astype(np.float64) * scale).astype(np.float32)
    pt_min_nt    = (pt_min.astype(np.float64) * scale).astype(np.float32)
    pt_max_nt    = (pt_max.astype(np.float64) * scale).astype(np.float32)
    pt_absmax_nt = (pt_absmax.astype(np.float64) * scale).astype(np.float32)

    mag_meas_nt = np.linalg.norm(mean_meas_nt, axis=1)
    mag_corr_nt = np.linalg.norm(mean_corr_nt, axis=1)

    i_min = int(np.argmin(mag_corr_nt))
    i_max = int(np.argmax(mag_corr_nt))

    map_stats = {
        "fs_actual": float(ni.actual_fs),
        "channels": [cfg.ch_x, cfg.ch_y, cfg.ch_z],
        "scale_nt_per_v": [float(x) for x in scale],
        "Bmag_corr_min_nT": float(mag_corr_nt[i_min]),
        "Bmag_corr_min_point": list(points[i_min]),
        "Bmag_corr_max_nT": float(mag_corr_nt[i_max]),
        "Bmag_corr_max_point": list(points[i_max]),
        "off0_tmid_s": float(off0_t),
        "off1_tmid_s": float(off1_t),
        "off0_offset_v": [float(x) for x in off0_vec],
        "off1_offset_v": [float(x) for x in off1_vec],
        "off0_qc": dict(off0.get("qc", {})),
        "off1_qc": dict(off1.get("qc", {})),
        "off0_steps": [
            {
                "label": s.get("label", ""),
                "start_idx": int(s.get("start_idx", -1)),
                "end_idx": int(s.get("end_idx", -1)),
                "tmid_s": float(s.get("tmid_s", np.nan)),
                "mean_v": [float(v) for v in np.asarray(s.get("mean_v", [np.nan, np.nan, np.nan])).reshape(3)],
            } for s in off0.get("steps", [])
        ],
        "off1_steps": [
            {
                "label": s.get("label", ""),
                "start_idx": int(s.get("start_idx", -1)),
                "end_idx": int(s.get("end_idx", -1)),
                "tmid_s": float(s.get("tmid_s", np.nan)),
                "mean_v": [float(v) for v in np.asarray(s.get("mean_v", [np.nan, np.nan, np.nan])).reshape(3)],
            } for s in off1.get("steps", [])
        ],
        "channels_field": getattr(ni, "field_channels", [cfg.ch_x, cfg.ch_y, cfg.ch_z]),
        "channels_aux": getattr(ni, "aux_channels", []),
        "aux_file": "stream_aux_f32.bin" if getattr(ni, "aux_n", 0) > 0 else None,
    }
    with open(os.path.join(session_dir, "map_stats.json"), "w") as f:
        json.dump(map_stats, f, indent=2)

    np.savez_compressed(
        os.path.join(session_dir, "map_data.npz"),
        points=np.array(points, dtype=np.int16),
        tmid_s=tmid_s,
        start_idx=start_idx_arr,
        end_idx=end_idx_arr,
        mean_meas_V=mean_meas,
        std_meas_V=std_meas,
        mean_corr_V=mean_corr_v,
        raw_points_V=raw_points,
        pt_min_V=pt_min,
        pt_max_V=pt_max,
        pt_absmax_V=pt_absmax,
        mean_meas_nT=mean_meas_nt,
        mean_corr_nT=mean_corr_nt,
        pt_min_nT=pt_min_nt,
        pt_max_nT=pt_max_nt,
        pt_absmax_nT=pt_absmax_nt,
        fs_actual=float(ni.actual_fs),
        qc_code=qc_code,
        qc_clip_frac=qc_clip_frac,
        off0_tmid_s=float(off0_t),
        off1_tmid_s=float(off1_t),
        off0_offset_v=off0_vec.astype(np.float32),
        off1_offset_v=off1_vec.astype(np.float32),
    )

    csv_path = os.path.join(session_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ix","iy","iz","x","y","z",
            "Bx_nT","By_nT","Bz_nT","Bmag_nT",
            "Bx_corr_nT","By_corr_nT","Bz_corr_nT","Bmag_corr_nT",
            "Bx_min_nT","By_min_nT","Bz_min_nT",
            "Bx_max_nT","By_max_nT","Bz_max_nT",
            "Bx_absmax_nT","By_absmax_nT","Bz_absmax_nT",
            "tmid_s","start_idx","end_idx",
            "qc_code","qc_clip_frac_x","qc_clip_frac_y","qc_clip_frac_z",
        ])

        for i, (ix, iy, iz) in enumerate(points):
            x, y, z = idx_to_xyz_m(ix, iy, iz, cfg.grid_n, cfg.grid_spacing_m)

            Bnt = mean_meas_nt[i]
            Cnt = mean_corr_nt[i]

            w.writerow([
                ix, iy, iz, x, y, z,
                float(Bnt[0]), float(Bnt[1]), float(Bnt[2]), float(np.linalg.norm(Bnt)),
                float(Cnt[0]), float(Cnt[1]), float(Cnt[2]), float(np.linalg.norm(Cnt)),
                float(pt_min_nt[i,0]), float(pt_min_nt[i,1]), float(pt_min_nt[i,2]),
                float(pt_max_nt[i,0]), float(pt_max_nt[i,1]), float(pt_max_nt[i,2]),
                float(pt_absmax_nt[i,0]), float(pt_absmax_nt[i,1]), float(pt_absmax_nt[i,2]),
                float(tmid_s[i]),
                int(start_idx_arr[i]), int(end_idx_arr[i]),
                int(qc_code[i]),
                float(qc_clip_frac[i,0]), float(qc_clip_frac[i,1]), float(qc_clip_frac[i,2]),
            ])

    export_cones(session_dir, cfg, points, mean_corr_nt, force_cmax=None,
             basename="cones_corr", title="Corrected field map (cones)")


# =========================
# Session workflow (offset wizard + point capture)
# =========================

class FieldMapSession:
    def __init__(self, cfg: AppConfig, ni: ContinuousNI, session_dir: str):
        self.cfg = cfg
        self.ni = ni
        self.session_dir = session_dir

        self.points = scan_points_min_movement(cfg.grid_n)
        self.point_idx = 0

        self.point_blocks = []
        self.point_stats = []

        self.aux_point_blocks = []
        self.aux_point_stats = []

        self.off0 = None
        self.off1 = None

        self.cal_mode = None  
        self.cal_step = 0
        self.cal_means = {}
        self.cal_steps_meta = []

    def next_point(self):
        if self.point_idx >= len(self.points):
            return None
        return self.points[self.point_idx]

    def start_wizard(self, mode: str):
        if mode not in ("start", "end"):
            raise ValueError("mode must be start or end")
        self.cal_mode = mode
        self.cal_step = 0
        self.cal_means = {}
        self.cal_steps_meta = []

    def cancel_wizard(self):
        self.cal_mode = None
        self.cal_step = 0
        self.cal_means = {}
        self.cal_steps_meta = []

    def wizard_next_label(self):
        if self.cal_mode is None:
            return None
        return self.cfg.flip_sequence[self.cal_step]

    def capture_wizard_step(self):
        if self.cal_mode is None:
            return False, "Wizard not active"

        label = self.cfg.flip_sequence[self.cal_step]
        ok = self.ni.request_capture("flip_offset", self.cfg.flip_capture_s)
        if not ok:
            return False, "Busy capturing, try again"
        cap = self.ni.wait_capture(timeout=10.0)
        if cap is None:
            return False, "Capture timeout"

        data = cap["data"] 
        mean_v = data.mean(axis=1).astype(np.float64)

        tmid = 0.5 * (cap["start_idx"] + cap["end_idx"]) / float(self.ni.actual_fs)

        self.cal_means[label] = mean_v
        self.cal_steps_meta.append({
            "label": label,
            "start_idx": int(cap["start_idx"]),
            "end_idx": int(cap["end_idx"]),
            "tmid_s": float(tmid),
            "mean_v": mean_v.astype(np.float32),
        })

        self.cal_step += 1
        if self.cal_step < len(self.cfg.flip_sequence):
            return True, f"Captured {label}. Next: {self.cfg.flip_sequence[self.cal_step]}"

        # wizard complete
        offset_v, field_v, qc = compute_offset_from_minmove_flips(self.cal_means)
        tmid_all = float(np.mean([m["tmid_s"] for m in self.cal_steps_meta]))

        result = {
            "method": "four_position_plate_side_method",
            "tmid_s": tmid_all,
            "offset_v": offset_v.astype(np.float64),
            "field_v": field_v.astype(np.float64),
            "qc": qc,
            "steps": self.cal_steps_meta,
        }

        if self.cal_mode == "start":
            self.off0 = result
            self.cancel_wizard()
            return True, "START offset done. Ready to record points."
        else:
            self.off1 = result
            self.cancel_wizard()
            return True, "END offset done. Press End Session to export."

    def record_point(self):
        if self.cal_mode is not None:
            return False, "Finish/cancel offset wizard first."
        if self.point_idx >= len(self.points):
            return False, "All points already captured."

        pt = self.points[self.point_idx]
        ok = self.ni.request_capture("point", self.cfg.point_capture_s)
        if not ok:
            return False, "Busy capturing, try again"
        cap = self.ni.wait_capture(timeout=10.0)
        if cap is None:
            return False, "Capture timeout"

        data = cap["data"].astype(np.float32, copy=False)
        st = qc_check_block(data, self.cfg)

        tmid = 0.5 * (cap["start_idx"] + cap["end_idx"]) / float(self.ni.actual_fs)
        st["tmid_s"] = float(tmid)

        blk = {
            "point": pt,
            "start_idx": int(cap["start_idx"]),
            "end_idx": int(cap["end_idx"]),
            "data": data,
        }

        self.point_blocks.append(blk)
        self.point_stats.append(st)
        self.point_idx += 1

        nxt = self.next_point()
        if nxt is None:
            return True, f"Captured {pt}. Next: DONE"
        return True, f"Captured {pt}. Next: {nxt}"
    
    def record_aux_point(self, comment: str = ""):
        if self.cal_mode is not None:
            return False, "Finish/cancel offset wizard first."

        ok = self.ni.request_capture("aux_point", self.cfg.point_capture_s)
        if not ok:
            return False, "Busy capturing, try again"

        cap = self.ni.wait_capture(timeout=10.0)
        if cap is None:
            return False, "Capture timeout"

        data = cap["data"].astype(np.float32, copy=False)
        st = qc_check_block(data, self.cfg)

        tmid = 0.5 * (cap["start_idx"] + cap["end_idx"]) / float(self.ni.actual_fs)
        st["tmid_s"] = float(tmid)

        mean_v = np.asarray(st["mean"], dtype=np.float64).reshape(3,)

        if self.off0 is not None and self.off1 is not None:
            off0_t = float(self.off0.get("tmid_s", tmid))
            off1_t = float(self.off1.get("tmid_s", off0_t))
            off0_vec = np.asarray(self.off0["offset_v"], dtype=np.float64).reshape(3,)
            off1_vec = np.asarray(self.off1["offset_v"], dtype=np.float64).reshape(3,)

            if off1_t <= off0_t + 1e-12:
                off_vec = off0_vec
            else:
                a = (tmid - off0_t) / (off1_t - off0_t)
                a = float(np.clip(a, 0.0, 1.0))
                off_vec = off0_vec + a * (off1_vec - off0_vec)

            mean_corr_v = mean_v - off_vec
        elif self.off0 is not None:
            off_vec = np.asarray(self.off0["offset_v"], dtype=np.float64).reshape(3,)
            mean_corr_v = mean_v - off_vec
        else:
            off_vec = np.zeros(3, dtype=np.float64)
            mean_corr_v = mean_v.copy()

        scale = np.asarray(self.cfg.scale_nt_per_v, dtype=np.float64).reshape(3,)
        mean_nt = mean_v * scale
        mean_corr_nt = mean_corr_v * scale
        min_nt = np.asarray(st["min"], dtype=np.float64).reshape(3,) * scale
        max_nt = np.asarray(st["max"], dtype=np.float64).reshape(3,) * scale
        absmax_nt = np.asarray(st["absmax"], dtype=np.float64).reshape(3,) * scale

        aux_blk = {
            "start_idx": int(cap["start_idx"]),
            "end_idx": int(cap["end_idx"]),
            "data": data,
            "comment": comment,
        }

        aux_st = {
            "mean": np.asarray(st["mean"], dtype=np.float32),
            "std": np.asarray(st["std"], dtype=np.float32),
            "min": np.asarray(st["min"], dtype=np.float32),
            "max": np.asarray(st["max"], dtype=np.float32),
            "absmax": np.asarray(st["absmax"], dtype=np.float32),
            "qc_code": int(st.get("qc_code", 0)),
            "qc_clip_frac": np.asarray(st.get("qc_clip_frac", [0, 0, 0]), dtype=np.float32),
            "tmid_s": float(tmid),

            "mean_corr_v": mean_corr_v.astype(np.float32),
            "mean_nt": mean_nt.astype(np.float32),
            "mean_corr_nt": mean_corr_nt.astype(np.float32),
            "min_nt": min_nt.astype(np.float32),
            "max_nt": max_nt.astype(np.float32),
            "absmax_nt": absmax_nt.astype(np.float32),
        }

        self.aux_point_blocks.append(aux_blk)
        self.aux_point_stats.append(aux_st)

        self._append_aux_point_csv(aux_blk, aux_st)

        return True, f"AUX point recorded. Comment: {comment if comment else '(blank)'}"
    
    def _append_aux_point_csv(self, blk: dict, st: dict):
        csv_path = os.path.join(self.session_dir, "aux_points.csv")
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            if not file_exists:
                w.writerow([
                    "ix","iy","iz","x","y","z",
                    "Bx_nT","By_nT","Bz_nT","Bmag_nT",
                    "Bx_corr_nT","By_corr_nT","Bz_corr_nT","Bmag_corr_nT",
                    "Bx_min_nT","By_min_nT","Bz_min_nT",
                    "Bx_max_nT","By_max_nT","Bz_max_nT",
                    "Bx_absmax_nT","By_absmax_nT","Bz_absmax_nT",
                    "tmid_s","start_idx","end_idx",
                    "qc_code","qc_clip_frac_x","qc_clip_frac_y","qc_clip_frac_z",
                    "comment"
                ])

            mean_nt = np.asarray(st["mean_nt"], dtype=np.float64).reshape(3,)
            mean_corr_nt = np.asarray(st["mean_corr_nt"], dtype=np.float64).reshape(3,)
            min_nt = np.asarray(st["min_nt"], dtype=np.float64).reshape(3,)
            max_nt = np.asarray(st["max_nt"], dtype=np.float64).reshape(3,)
            absmax_nt = np.asarray(st["absmax_nt"], dtype=np.float64).reshape(3,)
            clip = np.asarray(st["qc_clip_frac"], dtype=np.float64).reshape(3,)

            w.writerow([
                "", "", "", "", "", "",
                float(mean_nt[0]), float(mean_nt[1]), float(mean_nt[2]), float(np.linalg.norm(mean_nt)),
                float(mean_corr_nt[0]), float(mean_corr_nt[1]), float(mean_corr_nt[2]), float(np.linalg.norm(mean_corr_nt)),
                float(min_nt[0]), float(min_nt[1]), float(min_nt[2]),
                float(max_nt[0]), float(max_nt[1]), float(max_nt[2]),
                float(absmax_nt[0]), float(absmax_nt[1]), float(absmax_nt[2]),
                float(st["tmid_s"]),
                int(blk["start_idx"]), int(blk["end_idx"]),
                int(st["qc_code"]),
                float(clip[0]), float(clip[1]), float(clip[2]),
                blk.get("comment", ""),
            ])

    def undo_last_point(self):
        if len(self.point_blocks) == 0:
            return False, "No points to undo."
        self.point_blocks.pop()
        self.point_stats.pop()
        self.point_idx = max(0, self.point_idx - 1)
        return True, f"Undid last point. Next: {self.next_point()}"

    def retake_last_point(self):
        if len(self.point_blocks) == 0:
            return False, "No points to retake."
        self.point_idx = max(0, self.point_idx - 1)
        self.point_blocks.pop()
        self.point_stats.pop()
        return self.record_point()

    def export(self):
        if self.off0 is None or self.off1 is None:
            return False, "Need START and END offset before export."
        finalize_export(self.session_dir, self.cfg, self.ni, self.off0, self.off1, self.point_blocks, self.point_stats)
        return True, f"Exported session to: {self.session_dir}"
    
def export_partial_if_possible(session):
    """
    Export whatever is available.
    - If off0 & off1 exist and >=1 point: full corrected export (finalize_export)
    - Else: raw-only export (summary_raw.csv + map_stats.json note)
    Always writes run_complete flag.
    """
    cfg = session.cfg
    session_dir = session.session_dir

    n_points = len(session.point_blocks)
    complete = (n_points == len(session.points))

    if session.off0 is not None and session.off1 is not None and n_points > 0:
        finalize_export(
            session_dir, cfg, session.ni,
            session.off0, session.off1,
            session.point_blocks, session.point_stats
        )

        stats_path = os.path.join(session_dir, "map_stats.json")
        try:
            with open(stats_path, "r") as f:
                ms = json.load(f)
        except Exception:
            ms = {}

        ms["run_complete"] = bool(complete)
        ms["points_captured"] = int(n_points)
        ms["points_expected"] = int(len(session.points))
        ms["note"] = "complete" if complete else "incomplete run (user ended session early)"

        with open(stats_path, "w") as f:
            json.dump(ms, f, indent=2)

        return True, "Exported corrected map (full export)."

    if n_points == 0:
        ms = {
            "run_complete": False,
            "points_captured": 0,
            "points_expected": int(len(session.points)),
            "note": "no points captured; user ended session early",
            "has_off0": session.off0 is not None,
            "has_off1": session.off1 is not None,
        }
        with open(os.path.join(session_dir, "map_stats.json"), "w") as f:
            json.dump(ms, f, indent=2)
        return True, "No points to export (wrote map_stats.json only)."

    csv_path = os.path.join(session_dir, "summary_raw.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ix","iy","iz","x","y","z",
            "Bx_V","By_V","Bz_V","Bmag_V",
            "Bx_min_V","By_min_V","Bz_min_V",
            "Bx_max_V","By_max_V","Bz_max_V",
            "Bx_absmax_V","By_absmax_V","Bz_absmax_V",
            "tmid_s","start_idx","end_idx",
            "qc_code","qc_clip_frac_x","qc_clip_frac_y","qc_clip_frac_z",
        ])

        for blk, st in zip(session.point_blocks, session.point_stats):
            ix, iy, iz = blk["point"]
            x, y, z = idx_to_xyz_m(ix, iy, iz, cfg.grid_n, cfg.grid_spacing_m)

            mean = np.asarray(st["mean"], dtype=float)
            vmin = np.asarray(st["min"], dtype=float)
            vmax = np.asarray(st["max"], dtype=float)
            amax = np.asarray(st["absmax"], dtype=float)
            clip = st.get("qc_clip_frac", [0, 0, 0])

            w.writerow([
                ix, iy, iz, x, y, z,
                float(mean[0]), float(mean[1]), float(mean[2]), float(np.linalg.norm(mean)),
                float(vmin[0]), float(vmin[1]), float(vmin[2]),
                float(vmax[0]), float(vmax[1]), float(vmax[2]),
                float(amax[0]), float(amax[1]), float(amax[2]),
                float(st.get("tmid_s", np.nan)),
                int(blk["start_idx"]), int(blk["end_idx"]),
                int(st.get("qc_code", 0)),
                float(clip[0]), float(clip[1]), float(clip[2]),
            ])

    cones_note = None
    try:
        points = [blk["point"] for blk in session.point_blocks]
        mean_v = np.array([np.asarray(st["mean"], dtype=float).reshape(3) for st in session.point_stats], dtype=float)
        scale = np.asarray(cfg.scale_nt_per_v, dtype=float).reshape(3)
        mean_nt = (mean_v * scale[None, :]).astype(np.float32)

        export_cones(
            session_dir, cfg, points, mean_nt,
            basename="cones_raw",
            title="Raw field map (no offset) (cones)"
        )
        cones_note = "cones_raw.html"
    except Exception as e:
        print(f"[Export] Warning: raw cone export failed: {e}")

    ms = {
        "run_complete": bool(complete),
        "points_captured": int(n_points),
        "points_expected": int(len(session.points)),
        "note": "raw-only export (missing one or both offsets)",
        "has_off0": session.off0 is not None,
        "has_off1": session.off1 is not None,
        "scale_nt_per_v": [float(x) for x in np.asarray(cfg.scale_nt_per_v).reshape(3)],
        "cones_raw": cones_note,
    }

    with open(os.path.join(session_dir, "map_stats.json"), "w", encoding="utf-8") as f:
        json.dump(ms, f, indent=2)

    return True, f"Exported RAW-only summary to {csv_path}" + ("" if cones_note is None else f" + {cones_note}")
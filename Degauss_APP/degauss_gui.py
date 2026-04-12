from __future__ import annotations
import json
import sys
import time
import ipaddress
import socket
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
import redpitaya_scpi as scpi


# -----------------------------
# Data model
# -----------------------------
@dataclass
class DegaussParams:
    rp_ip: str = "169.254.102.243"
    out_mode: str = "OUT1"                
    f0_hz: float = 7.0
    periods_up: int = 10
    periods_hold: int = 10
    periods_down: int = 1500
    amp_vpp: float = 1.0            
    envelope: str = "linear"          
    log_decades: float = 3.0          
    log_dir: str = "degauss_logs"     

    @property
    def t0_s(self) -> float:
        return 1.0 / float(self.f0_hz)

    @property
    def total_periods(self) -> int:
        return int(self.periods_up + self.periods_hold + self.periods_down)

    @property
    def total_time_s(self) -> float:
        return self.total_periods * self.t0_s

    @property
    def amp_vpeak(self) -> float:
        return float(self.amp_vpp) / 2.0


def build_envelope_per_period(p: DegaussParams) -> np.ndarray:
    """Per-period amplitude envelope in Vpeak: ramp up -> hold -> ramp down."""
    A = p.amp_vpeak
    Nu, Nh, Nd = p.periods_up, p.periods_hold, p.periods_down

    if Nu <= 0 or Nh < 0 or Nd <= 0:
        raise ValueError("Periods must satisfy: up>0, hold>=0, down>0")

    if p.envelope == "linear":
        env_up = A * (np.arange(1, Nu + 1, dtype=float) / Nu)
        env_hold = np.full(Nh, A, dtype=float)
        env_down = A * (1.0 - (np.arange(1, Nd + 1, dtype=float) / Nd))
        env_down[-1] = 0.0
    elif p.envelope == "log":
        d = max(0.1, float(p.log_decades))
        up = np.logspace(-d, 0.0, Nu, base=10.0)
        down = np.logspace(0.0, -d, Nd, base=10.0)
        env_up = A * (up / up.max())
        env_hold = np.full(Nh, A, dtype=float)
        env_down = A * (down / down.max())
        env_down[-1] = 0.0
    else:
        raise ValueError("Envelope must be 'linear' or 'log'")

    env = np.concatenate([env_up, env_hold, env_down])
    if env.size != p.total_periods:
        raise RuntimeError("Envelope length mismatch")
    return env


# -----------------------------
# RP control helpers
# -----------------------------
def rp_setup_sine(rp: scpi.scpi, mode: str, f0: float) -> None:
    rp.tx_txt("GEN:RST")

    def setup_ch(ch: int):
        rp.tx_txt(f"SOUR{ch}:FUNC SINE")
        rp.tx_txt(f"SOUR{ch}:FREQ:FIX {f0}")
        rp.tx_txt(f"SOUR{ch}:VOLT 0")
        rp.tx_txt(f"SOUR{ch}:VOLT:OFFS 0")
        rp.tx_txt(f"SOUR{ch}:TRIG:SOUR INT")

    rp.tx_txt("SOUR1:VOLT 0")
    rp.tx_txt("SOUR2:VOLT 0")
    rp.tx_txt("OUTPUT1:STATE OFF")
    rp.tx_txt("OUTPUT2:STATE OFF")

    if mode == "OUT1":
        setup_ch(1)
        rp.tx_txt("OUTPUT1:STATE ON")

    elif mode == "OUT2":
        setup_ch(2)
        rp.tx_txt("OUTPUT2:STATE ON")

    elif mode == "BOTH":
        setup_ch(1)
        setup_ch(2)
        rp.tx_txt("OUTPUT1:STATE ON")
        rp.tx_txt("OUTPUT2:STATE ON")

    else:
        raise ValueError("mode must be OUT1, OUT2, or BOTH")
    
def emergency_rp_stop(ip: str) -> None:
    """Best-effort immediate shutdown using a fresh SCPI connection."""
    try:
        rp = scpi.scpi(ip)
        rp_stop_all(rp)
        try:
            rp.close()
        except Exception:
            pass
    except Exception:
        pass


def _is_valid_ipv4(ip_str: str) -> bool:
    try:
        ipaddress.ip_address(ip_str)
        return True
    except ValueError:
        return False


def _probe_tcp(ip: str, port: int, timeout_s: float) -> bool:
    try:
        with closing(socket.create_connection((ip, port), timeout=timeout_s)):
            return True
    except OSError:
        return False


def preflight_rp_scpi(ip: str, timeout_s: float = 0.6) -> tuple[bool, str]:
    """
    Quick check to fail fast on wrong IP.
    Tries typical Red Pitaya SCPI ports. (Most common is 5000.)
    """
    ip = ip.strip()
    if not _is_valid_ipv4(ip):
        return False, "Invalid IPv4 address format."

    ports_to_try = [5000, 5025]  
    for port in ports_to_try:
        if _probe_tcp(ip, port, timeout_s):
            return True, f"SCPI port reachable (TCP {port})."

    return False, f"Could not reach SCPI server on {ip} (tried ports {ports_to_try}). Check IP/cable/RP SCPI server."


def rp_stop_all(rp: scpi.scpi) -> None:
    rp.tx_txt("SOUR1:VOLT 0")
    rp.tx_txt("SOUR2:VOLT 0")
    rp.tx_txt("OUTPUT1:STATE OFF")
    rp.tx_txt("OUTPUT2:STATE OFF")


# -----------------------------
# Worker thread
# -----------------------------
class DegaussWorker(QtCore.QObject):
    progress = QtCore.Signal(float, float, float)  
    finished = QtCore.Signal(bool, str, dict)      

    def __init__(self, params: DegaussParams):
        super().__init__()
        self.params = params
        self._stop_requested = False
        self._rp = None

    @QtCore.Slot()
    def run(self) -> None:
        p = self.params
        env = build_envelope_per_period(p)

        run_info = {
            "params": asdict(p),
            "planned_total_time_s": p.total_time_s,
            "planned_total_periods": p.total_periods,
            "started_epoch_s": time.time(),
        }

        rp = scpi.scpi(p.rp_ip)
        self._rp = rp
        ok = False
        msg = "Unknown"

        try:
            rp_setup_sine(rp, p.out_mode, p.f0_hz)

            if p.out_mode in ("OUT1", "BOTH"):
                rp.tx_txt("SOUR1:TRIG:INT")
            if p.out_mode in ("OUT2", "BOTH"):
                rp.tx_txt("SOUR2:TRIG:INT")


            start = time.perf_counter()
            t0 = p.t0_s
            N = p.total_periods

            for i, amp_peak in enumerate(env):
                if self._stop_requested:
                    msg = "Stopped by user"
                    ok = False
                    break

                t_target = start + i * t0
                while True:
                    if self._stop_requested:
                        break
                    remaining = t_target - time.perf_counter()
                    if remaining <= 0:
                        break
                    time.sleep(min(0.05, remaining))  
                
                if self._stop_requested:
                    break



                if p.out_mode == "OUT1":
                    rp.tx_txt(f"SOUR1:VOLT {amp_peak:.6f}")
                elif p.out_mode == "OUT2":
                    rp.tx_txt(f"SOUR2:VOLT {amp_peak:.6f}")
                elif p.out_mode == "BOTH":
                    rp.tx_txt(f"SOUR1:VOLT {amp_peak:.6f}")
                    rp.tx_txt(f"SOUR2:VOLT {amp_peak:.6f}")


                elapsed = time.perf_counter() - start
                frac = (i + 1) / N
                self.progress.emit(frac, elapsed, float(amp_peak))

            else:
                ok = True
                msg = "Run completed"

        except Exception as e:
            ok = False
            msg = f"Error: {e!r}"

        finally:
            try:
                rp_stop_all(rp)
            except Exception:
                pass
            try:
                rp.close()
            except Exception:
                pass
            self._rp = None

            run_info["ended_epoch_s"] = time.time()
            run_info["ok"] = ok
            run_info["message"] = msg

            self.finished.emit(ok, msg, run_info)

    def request_stop(self) -> None:
        self._stop_requested = True

    def stop_now(self):
        self._stop_requested = True
        try:
            if self._rp is not None:
                rp_stop_all(self._rp)  
        except Exception:
            pass


# -----------------------------
# GUI
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    stopRequested = QtCore.Signal()
    def __init__(self):
        super().__init__()

        self._applying = False 

        self.setWindowTitle("Degaussing Control (RP SCPI)")

        self.defaults = DegaussParams()
        self.params = DegaussParams()
        self.worker = None
        self.thread = None

        self._build_ui()
        self._apply_params_to_widgets()
        self._update_preview()


    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QFrame()
        left.setFrameShape(QtWidgets.QFrame.StyledPanel)
        left_layout = QtWidgets.QVBoxLayout(left)

        grp_setup = QtWidgets.QGroupBox("Setup")
        form = QtWidgets.QFormLayout(grp_setup)

        self.ip = QtWidgets.QLineEdit()
        self.out_sel = QtWidgets.QComboBox()
        self.out_sel.addItems(["OUT1", "OUT2", "OUT1+OUT2"])


        self.f0 = QtWidgets.QDoubleSpinBox()
        self.f0.setRange(0.1, 200.0)
        self.f0.setDecimals(3)
        self.f0.setSingleStep(0.1)

        self.amp = QtWidgets.QDoubleSpinBox()
        self.amp.setRange(0.0, 5.0)  
        self.amp.setDecimals(3)
        self.amp.setSingleStep(0.1)
        self.amp.setSuffix(" Vpp")

        self.up = QtWidgets.QSpinBox()
        self.up.setRange(1, 3000)

        self.hold = QtWidgets.QSpinBox()
        self.hold.setRange(0, 3000)

        self.down = QtWidgets.QSpinBox()
        self.down.setRange(1, 3000)



        self.env = QtWidgets.QComboBox()
        self.env.addItems(["linear", "log"])
        self.log_dec = QtWidgets.QDoubleSpinBox()
        self.log_dec.setRange(0.1, 10.0)
        self.log_dec.setDecimals(2)
        self.log_dec.setSingleStep(0.25)
        self.log_dec.setSuffix(" decades")

        self.log_dir = QtWidgets.QLineEdit()

        self.lbl_time = QtWidgets.QLabel("—")
        self.lbl_peak = QtWidgets.QLabel("—")

        form.addRow("RP IP:", self.ip)
        form.addRow("Output:", self.out_sel)
        form.addRow("Frequency:", self.f0)
        form.addRow("Amplitude:", self.amp)
        form.addRow("Ramp-up (periods):", self.up)
        form.addRow("Hold (periods):", self.hold)
        form.addRow("Ramp-down (periods):", self.down)
        form.addRow("Envelope:", self.env)
        form.addRow("Log decades:", self.log_dec)
        form.addRow("Log folder:", self.log_dir)
        form.addRow("Total time:", self.lbl_time)
        form.addRow("Max (Vpeak):", self.lbl_peak)

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("STOP")
        self.btn_reset = QtWidgets.QPushButton("Reset defaults")

        self.btn_stop.setEnabled(False)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)

        left_layout.addWidget(grp_setup)
        left_layout.addLayout(btn_row)
        left_layout.addWidget(self.btn_reset)
        left_layout.addStretch(1)

        right = QtWidgets.QFrame()
        right.setFrameShape(QtWidgets.QFrame.StyledPanel)
        right_layout = QtWidgets.QVBoxLayout(right)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "t", units="s")
        self.plot.setLabel("left", "Amplitude", units="V (arb preview)")
        self.curve = self.plot.plot([], [], pen=pg.mkPen(width=2))
        self.marker = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen(width=2))
        self.plot.addItem(self.marker)

        self.status = QtWidgets.QLabel("Idle.")
        self.pbar = QtWidgets.QProgressBar()
        self.pbar.setRange(0, 100)

        right_layout.addWidget(self.plot, 1)
        right_layout.addWidget(self.pbar)
        right_layout.addWidget(self.status)

        layout.addWidget(left, 0)
        layout.addWidget(right, 1)

        for w in [self.ip, self.out_sel, self.f0, self.amp, self.up, self.hold, self.down, self.env, self.log_dec, self.log_dir]:
            if isinstance(w, QtWidgets.QLineEdit):
                w.textChanged.connect(self._on_params_changed)
            else:
                w.valueChanged.connect(self._on_params_changed) if hasattr(w, "valueChanged") else w.currentIndexChanged.connect(self._on_params_changed)

        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_reset.clicked.connect(self._reset)

        for sb in (self.up, self.hold, self.down):
            sb.setKeyboardTracking(False)

        for dsb in (self.f0, self.amp, self.log_dec):
            dsb.setKeyboardTracking(False)

    def _apply_params_to_widgets(self):
        p = self.params
        self._applying = True
        try:
            self.ip.setText(p.rp_ip)
            self.out_sel.setCurrentIndex(0 if p.out_mode == "OUT1" else (1 if p.out_mode == "OUT2" else 2))
            self.f0.setValue(p.f0_hz)
            self.amp.setValue(p.amp_vpp)
            self.up.setValue(p.periods_up)
            self.hold.setValue(p.periods_hold)
            self.down.setValue(p.periods_down)
            self.env.setCurrentText(p.envelope)
            self.log_dec.setValue(p.log_decades)
            self.log_dir.setText(p.log_dir)
        finally:
            self._applying = False

        self._refresh_labels()


    def _read_widgets_to_params(self):
        p = self.params
        p.rp_ip = self.ip.text().strip()
        p.out_channel = 1 if self.out_sel.currentIndex() == 0 else 2
        p.f0_hz = float(self.f0.value())
        p.amp_vpp = float(self.amp.value())
        p.periods_up = int(self.up.value())
        p.periods_hold = int(self.hold.value())
        p.periods_down = int(self.down.value())
        p.envelope = self.env.currentText()
        p.log_decades = float(self.log_dec.value())
        p.log_dir = self.log_dir.text().strip() or "degauss_logs"
        idx = self.out_sel.currentIndex()
        self.params.out_mode = "OUT1" if idx == 0 else ("OUT2" if idx == 1 else "BOTH")


    def _refresh_labels(self):
        p = self.params
        self.lbl_time.setText(f"{p.total_time_s:.1f} s  (~{p.total_time_s/60:.2f} min)")
        self.lbl_peak.setText(f"{p.amp_vpeak:.3f} V")

        self.log_dec.setEnabled(p.envelope == "log")

    def _on_params_changed(self, *_):
        if self._applying:
            return
        self._read_widgets_to_params()
        self._refresh_labels()
        self._update_preview()
        ok, reason = preflight_rp_scpi(self.params.rp_ip, timeout_s=0.6)
        if not ok:
            QtWidgets.QMessageBox.critical(
                self,
                "Connection error",
                f"Couldn't connect to Red Pitaya at {self.params.rp_ip}.\n\n{reason}\n\nResetting to defaults."
            )
            self.status.setText(f"Connection failed: {reason}")
            self._reset()
            return



    def _update_preview(self):
        p = self.params
        try:
            env = build_envelope_per_period(p)
            T = p.total_time_s
            f0 = p.f0_hz

            max_samples = 200_000
            spp = 200
            n = min(int(p.total_periods * spp), max_samples)
            t = np.linspace(0.0, T, n, endpoint=False)

            idx = np.minimum((t / p.t0_s).astype(int), p.total_periods - 1)
            amp = env[idx]
            y = amp * np.sin(2 * np.pi * f0 * t)

            self.curve.setData(t, y)
            self.marker.setPos(0.0)
            self.status.setText("Idle.")
            self.pbar.setValue(0)

        except Exception as e:
            self.curve.setData([], [])
            self.status.setText(f"Parameter error: {e}")
            self.pbar.setValue(0)

    def _start(self):
        if self.thread is not None:
            return

        self._read_widgets_to_params()

        if not self.params.rp_ip:
            self.status.setText("RP IP is empty.")
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText("Starting...")

        self.thread = QtCore.QThread()
        self.worker = DegaussWorker(self.params)
        self.worker.moveToThread(self.thread)
        self.stopRequested.connect(self.worker.stop_now, QtCore.Qt.QueuedConnection)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self._cleanup_thread)

        self.thread.start()

    def _stop(self):
        if self.worker:
            self.worker.request_stop()

        emergency_rp_stop(self.params.rp_ip)

        self.status.setText("Stopping (emergency OFF sent)...")
        self.btn_stop.setEnabled(False)



    def _on_progress(self, frac: float, elapsed_s: float, amp_vpeak: float):
        p = self.params
        self.pbar.setValue(int(frac * 100))
        self.marker.setPos(min(elapsed_s, p.total_time_s))
        remaining = max(0.0, p.total_time_s - elapsed_s)
        self.status.setText(
            f"Running… {frac*100:5.1f}% | elapsed {elapsed_s:6.1f}s | remaining {remaining:6.1f}s | amp {amp_vpeak:.3f} Vpeak"
        )

    def _on_finished(self, ok: bool, message: str, run_info: dict):
        self.status.setText(message)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        try:
            base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
            base_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
            log_dir = (base_dir / self.params.log_dir).resolve()
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out = log_dir / f"degauss_{ts}_{'OK' if ok else 'FAIL'}.json"
            out.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
        except Exception as e:
            self.status.setText(f"{message} (log save failed: {e})")

    def _cleanup_thread(self):
        try:
            if self.worker:
                self.stopRequested.disconnect(self.worker.stop_now)
        except Exception:
            pass
        self.thread = None
        self.worker = None


    def _reset(self):
        self.params = DegaussParams(**asdict(self.defaults))
        self._apply_params_to_widgets()
        self._update_preview()


    def closeEvent(self, event):
        try:
            self._stop()
        except Exception:
            pass

        if self.thread:
            self.thread.quit()
            self.thread.wait(2000)

        event.accept()




def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = MainWindow()
    w.resize(1200, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

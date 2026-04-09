import os
import sys
import numpy as np
import json
from pathlib import Path
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QLineEdit, QFileDialog,
    QMessageBox, QGroupBox, QCheckBox, QInputDialog,
    QToolButton, QScrollArea, QSizePolicy, QSplitter,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import core

def _settings_path():
    # Store settings next to the executable (portable), or next to the script when running from source
    try:
        exe_dir = Path(sys.executable).resolve().parent
    except Exception:
        exe_dir = Path(__file__).resolve().parent
    return exe_dir / "user_settings.json"


def load_user_settings():
    p = _settings_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_user_settings(d: dict):
    p = _settings_path()
    try:
        p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass

print("USING core from:", core.__file__)

class CollapsibleBox(QWidget):
    def __init__(self, title: str, collapsed: bool = True, parent=None):
        super().__init__(parent)

        self.toggle = QToolButton(text=title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(not collapsed)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow if not collapsed else Qt.RightArrow)
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle.toggled.connect(self._on_toggled)

        self.content = QWidget()
        self.content.setVisible(not collapsed)
        self.content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.toggle)
        lay.addWidget(self.content)

    def setContentLayout(self, layout):
        self.content.setLayout(layout)

    def _on_toggled(self, checked: bool):
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)

    def set_collapsed(self, collapsed: bool):
        self.toggle.setChecked(not collapsed)



class MonitorCanvas(FigureCanvas):
    def __init__(self, labels):
        self.labels = list(labels)
        n_ch = len(self.labels)

        fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = [fig.add_subplot(n_ch, 1, k + 1) for k in range(n_ch)]
        super().__init__(fig)
        fig.subplots_adjust(left=0.16, right=0.985, top=0.97, bottom=0.10, hspace=0.28)

        self.lines = []
        for k, a in enumerate(self.ax):
            a.grid(True)
            a.set_xlim(0, 10)
            a.set_ylabel(self.labels[k])
            a.yaxis.labelpad = 8
            (ln,) = a.plot([], [], lw=1)
            self.lines.append(ln)

        self.ax[-1].set_xlabel("t (s)")

    def set_ylabels(self, unit: str, mode: str):
        for k, a in enumerate(self.ax):
            a.set_ylabel(f"{self.labels[k]} ({unit}, {mode})")

    def update_data(self, t_s: np.ndarray, data_nxn: np.ndarray):
        # data_nxn in display units, shape (n_ch, N)
        n_ch = min(len(self.labels), data_nxn.shape[0])
        for k in range(n_ch):
            y = data_nxn[k]
            self.lines[k].set_data(t_s, y)
            self.ax[k].set_xlim(float(t_s[0]), float(t_s[-1]))

            # autoscale y with small padding
            y_min = float(np.min(y))
            y_max = float(np.max(y))
            if np.isfinite(y_min) and np.isfinite(y_max):
                if abs(y_max - y_min) < 1e-12:
                    pad = 1e-3 if abs(y_min) < 1e-6 else 0.05 * abs(y_min)
                    self.ax[k].set_ylim(y_min - pad, y_max + pad)
                else:
                    pad = 0.1 * (y_max - y_min)
                    self.ax[k].set_ylim(y_min - pad, y_max + pad)

        self.draw_idle()



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Field Map Acquisition (NI)")

        self.cfg = core.AppConfig()
        self.ni = None
        self.session = None
        self.session_running = False

        # UI state
        self.show_nt = False
        self.show_corr = False
        self.paused = False
        self.show_aux = False

        # ------- UI layout -------
        root = QWidget()
        self.setCentralWidget(root)

        main = QHBoxLayout(root)

        main.setContentsMargins(0, 0, 0, 0)

        left_container = QWidget()
        left = QVBoxLayout(left_container)
        left.setContentsMargins(6, 6, 6, 6)
        left.setSpacing(8)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_container)

        # --- Splitter ensures sidebar never overlaps plots ---
        split = QSplitter(Qt.Horizontal)
        main.addWidget(split)

        # Left sidebar (scrollable)
        left_scroll.setMinimumWidth(380)    
        left_scroll.setMaximumWidth(650)   
        split.addWidget(left_scroll)

        # Right area in an explicit widget (prevents overlap issues)
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(6, 6, 6, 6)
        right.setSpacing(6)
        split.addWidget(right_widget)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setCollapsible(0, False)
        split.setHandleWidth(8)

        # --- Vertical splitter inside the right pane for Field vs AUX monitors ---
        self.monitor_split = QSplitter(Qt.Vertical)
        self.monitor_split.setHandleWidth(8)
        self.monitor_split.setChildrenCollapsible(False)
        right.addWidget(self.monitor_split, 1)

        # Monitor (Field XYZ)
        self.canvas_field = MonitorCanvas(["X", "Y", "Z"])
        self.canvas_field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.monitor_split.addWidget(self.canvas_field)

        # Monitor (AUX) - hidden by default
        self.canvas_aux = MonitorCanvas(["AUX1", "AUX2", "AUX3"])
        self.canvas_aux.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_aux.setVisible(False)
        self.monitor_split.addWidget(self.canvas_aux)

        # Initial vertical split ratio: mostly field, smaller AUX
        self.monitor_split.setStretchFactor(0, 4)
        self.monitor_split.setStretchFactor(1, 2)
        self.monitor_split.setSizes([700, 250])

        # Status
        self.status = QLabel("Idle.")
        right.addWidget(self.status, 0)
        
        # ---- Setup box ----
        self.box_setup = CollapsibleBox("Setup", collapsed=True)
        left.addWidget(self.box_setup)
        g = QGridLayout()
        self.box_setup.setContentLayout(g)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["NI-DAQmx"])
        g.addWidget(QLabel("Backend:"), 0, 0)
        g.addWidget(self.backend_combo, 0, 1)

        self.dev_combo = QComboBox()
        g.addWidget(QLabel("Field NI device:"), 1, 0)
        g.addWidget(self.dev_combo, 1, 1)

        # NEW: separate module/device selector for AUX channels
        self.aux_dev_combo = QComboBox()
        g.addWidget(QLabel("AUX NI device:"), 2, 0)
        g.addWidget(self.aux_dev_combo, 2, 1)

        self.chx_combo = QComboBox()
        self.chy_combo = QComboBox()
        self.chz_combo = QComboBox()
        g.addWidget(QLabel("Ch X:"), 3, 0); g.addWidget(self.chx_combo, 3, 1)
        g.addWidget(QLabel("Ch Y:"), 4, 0); g.addWidget(self.chy_combo, 4, 1)
        g.addWidget(QLabel("Ch Z:"), 5, 0); g.addWidget(self.chz_combo, 5, 1)

        # --- AUX UI (now 3 raw voltage channels) ---
        self.aux_enable = QCheckBox("Enable AUX logging")
        self.aux_enable.setChecked(False)
        g.addWidget(self.aux_enable, 6, 0, 1, 2)

        self.aux1_combo = QComboBox()
        self.aux2_combo = QComboBox()
        self.aux3_combo = QComboBox()
        g.addWidget(QLabel("AUX1:"), 7, 0); g.addWidget(self.aux1_combo, 7, 1)
        g.addWidget(QLabel("AUX2:"), 8, 0); g.addWidget(self.aux2_combo, 8, 1)
        g.addWidget(QLabel("AUX3:"), 9, 0); g.addWidget(self.aux3_combo, 9, 1)

        self.fs_box = QDoubleSpinBox()
        self.fs_box.setRange(10, 100000)
        self.fs_box.setValue(self.cfg.fs_req)
        self.fs_box.setDecimals(1)
        g.addWidget(QLabel("fs (Hz):"), 10, 0); g.addWidget(self.fs_box, 10, 1)

        self.scale_box = QDoubleSpinBox()
        self.scale_box.setRange(0.1, 1e9)
        self.scale_box.setValue(float(self.cfg.scale_nt_per_v[0]))
        self.scale_box.setDecimals(1)
        g.addWidget(QLabel("nT/V scale:"), 11, 0); g.addWidget(self.scale_box, 11, 1)

        settings = load_user_settings()
        self.out_edit = QLineEdit(settings.get("last_save_root", ""))
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self.pick_out_dir)
        g.addWidget(QLabel("Save root:"), 12, 0); g.addWidget(self.out_edit, 12, 1)
        g.addWidget(btn_out, 13, 1)

        self.btn_refresh = QPushButton("Refresh Devices")
        self.btn_refresh.clicked.connect(self.refresh_devices)
        g.addWidget(self.btn_refresh, 14, 1)

        # disable AUX controls when AUX logging is off
        def _aux_ui_update():
            en = self.aux_enable.isChecked()
            self.aux_dev_combo.setEnabled(en)
            self.aux1_combo.setEnabled(en)
            self.aux2_combo.setEnabled(en)
            self.aux3_combo.setEnabled(en)

        self.aux_enable.stateChanged.connect(_aux_ui_update)
        _aux_ui_update()

        # IMPORTANT: connect device changes to refresh channels (do this once)
        self.dev_combo.currentTextChanged.connect(self.refresh_channels)
        self.aux_dev_combo.currentTextChanged.connect(self.refresh_channels)

        # ---- Controls box ----
        self.box_meas = CollapsibleBox("Measurement", collapsed=True)
        left.addWidget(self.box_meas)
        c = QVBoxLayout()
        self.box_meas.setContentLayout(c)

        self.btn_start = QPushButton("Start Session")
        self.btn_start.clicked.connect(self.start_session)
        c.addWidget(self.btn_start)

        # --- Next point label + tiny 3D grid preview ---
        row_next = QHBoxLayout()
        self.lbl_next = QLabel("Next: (start session)")
        row_next.addWidget(self.lbl_next, stretch=1)

        self.grid_fig = Figure(figsize=(2.2, 2.2))
        self.grid_canvas = FigureCanvas(self.grid_fig)
        self.grid_canvas.setFixedSize(220, 220)  # tiny preview
        self.grid_ax = self.grid_fig.add_subplot(111, projection="3d")

        row_next.addWidget(self.grid_canvas, stretch=0)
        c.addLayout(row_next)

        # precompute 5x5x5 coordinates once (uses cfg.grid_n)
        n = int(self.cfg.grid_n)
        xs, ys, zs = np.meshgrid(np.arange(1, n+1), np.arange(1, n+1), np.arange(1, n+1), indexing="xy")
        self._grid_coords = (xs.ravel(), ys.ravel(), zs.ravel())

        self._draw_grid_preview(None)  # initial blank

        row1 = QHBoxLayout()
        self.btn_off0 = QPushButton("Start Offset Wizard")
        self.btn_step = QPushButton("Capture Step")
        self.btn_cancel = QPushButton("Cancel Wizard")
        row1.addWidget(self.btn_off0)
        row1.addWidget(self.btn_step)
        row1.addWidget(self.btn_cancel)
        c.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_point = QPushButton("Record Point")
        self.btn_aux_point = QPushButton("AUX Point")
        self.btn_undo = QPushButton("Undo")
        self.btn_retake = QPushButton("Retake")
        row2.addWidget(self.btn_point)
        row2.addWidget(self.btn_aux_point)
        row2.addWidget(self.btn_undo)
        row2.addWidget(self.btn_retake)
        c.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_off1 = QPushButton("Record END offset")
        row3.addWidget(self.btn_off1)
        c.addLayout(row3)

        # ---- Monitor toggles ----
        self.box_mon = CollapsibleBox("Monitor", collapsed=True)
        left.addWidget(self.box_mon)
        m = QVBoxLayout()
        self.box_mon.setContentLayout(m)
        left.addStretch(1)

        r = QHBoxLayout()
        self.chk_nt = QCheckBox("Show nT")
        self.chk_corr = QCheckBox("Show Corr (START)")
        self.chk_pause = QCheckBox("Pause")
        self.chk_aux = QCheckBox("Show AUX")

        r.addWidget(self.chk_nt)
        r.addWidget(self.chk_corr)
        r.addWidget(self.chk_pause)
        r.addWidget(self.chk_aux)
        m.addLayout(r)

        btn_clear = QPushButton("Clear Plot")
        btn_clear.clicked.connect(self.clear_plot)
        m.addWidget(btn_clear)

        # Bind
        self.btn_off0.clicked.connect(self.start_off0)
        self.btn_step.clicked.connect(self.capture_step)
        self.btn_cancel.clicked.connect(self.cancel_wizard)
        self.btn_point.clicked.connect(self.record_point)
        self.btn_aux_point.clicked.connect(self.record_aux_point)
        self.btn_undo.clicked.connect(self.undo_point)
        self.btn_retake.clicked.connect(self.retake_point)
        self.btn_off1.clicked.connect(self.start_off1_export)

        self.chk_nt.stateChanged.connect(self.toggle_units)
        self.chk_corr.stateChanged.connect(self.toggle_corr)
        self.chk_pause.stateChanged.connect(self.toggle_pause)
        self.chk_aux.stateChanged.connect(self.toggle_aux)

        self.enable_controls(False)

        # Timer update for monitor
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_monitor)
        self.timer.start(50)

        self.refresh_devices()

    # ---------------- UI helpers ----------------

    def enable_controls(self, enabled: bool):
        self.btn_off0.setEnabled(enabled)
        self.btn_step.setEnabled(enabled)
        self.btn_cancel.setEnabled(enabled)
        self.btn_point.setEnabled(enabled)
        self.btn_aux_point.setEnabled(enabled)
        self.btn_undo.setEnabled(enabled)
        self.btn_retake.setEnabled(enabled)
        self.btn_off1.setEnabled(enabled)
        self.chk_nt.setEnabled(enabled)
        self.chk_corr.setEnabled(enabled)
        self.chk_pause.setEnabled(enabled)

    def show_error(self, title, text):
        QMessageBox.critical(self, title, text)

    def pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select save root", self.out_edit.text())
        if d:
            self.out_edit.setText(d)

    def refresh_devices(self):
        if not core.NI_OK:
            details = ""
            if getattr(core, "NI_IMPORT_ERROR", None):
                details += f"NI_IMPORT_ERROR: {core.NI_IMPORT_ERROR}\n\n"
            if getattr(core, "NI_DIAG", None):
                details += "NI_DIAG:\n"
                for k, v in core.NI_DIAG.items():
                    details += f"  {k}: {v}\n"

            self.show_error(
                "NI backend not available",
                "NI Python support is not working.\n\n"
                "This usually means:\n"
                "- nidaqmx Python package is not bundled into the exe, OR\n"
                "- NI-DAQmx driver DLL (nicaiu) is missing / wrong bitness.\n\n"
                + details
            )
            return

        # keep previous selections BEFORE clearing
        prev_field = self.dev_combo.currentText()
        prev_aux = self.aux_dev_combo.currentText()

        devs = core.list_ni_devices()
        if not devs:
            devs = ["(none)"]

        self.dev_combo.blockSignals(True)
        self.aux_dev_combo.blockSignals(True)

        self.dev_combo.clear()
        self.aux_dev_combo.clear()
        self.dev_combo.addItems(devs)
        self.aux_dev_combo.addItems(devs)

        if prev_field in devs:
            self.dev_combo.setCurrentText(prev_field)
        if prev_aux in devs:
            self.aux_dev_combo.setCurrentText(prev_aux)

        self.dev_combo.blockSignals(False)
        self.aux_dev_combo.blockSignals(False)

        self.refresh_channels()

    def refresh_channels(self):
        field_dev = self.dev_combo.currentText()
        aux_dev = self.aux_dev_combo.currentText()

        field_chans = core.list_ai_channels(field_dev) or [self.cfg.ch_x, self.cfg.ch_y, self.cfg.ch_z]

        # ensure AUX defaults exist
        if not hasattr(self.cfg, "ch_aux1"):
            self.cfg.ch_aux1 = "cDAQ1Mod3/ai0"
        if not hasattr(self.cfg, "ch_aux2"):
            self.cfg.ch_aux2 = "cDAQ1Mod3/ai1"
        if not hasattr(self.cfg, "ch_aux3"):
            self.cfg.ch_aux3 = "cDAQ1Mod3/ai2"

        aux_chans = core.list_ai_channels(aux_dev) or [self.cfg.ch_aux1, self.cfg.ch_aux2, self.cfg.ch_aux3]

        def fill(combo, items, default):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            if default in items:
                combo.setCurrentText(default)
            combo.blockSignals(False)

        fill(self.chx_combo, field_chans, self.cfg.ch_x)
        fill(self.chy_combo, field_chans, self.cfg.ch_y)
        fill(self.chz_combo, field_chans, self.cfg.ch_z)

        fill(self.aux1_combo, aux_chans, self.cfg.ch_aux1)
        fill(self.aux2_combo, aux_chans, self.cfg.ch_aux2)
        fill(self.aux3_combo, aux_chans, self.cfg.ch_aux3)

    # ---------------- session control ----------------

    def start_session(self):

        # Toggle behavior
        if self.session_running:
            self.end_session()
            return

        root = self.out_edit.text().strip()

        # If empty or invalid -> ask user
        if (not root) or (not os.path.isdir(root)):
            root = QFileDialog.getExistingDirectory(self, "Select Save Root Folder")
            if not root:
                self.status.setText("Start canceled (no save folder selected).")
                return
            self.out_edit.setText(root)

        # Ensure it exists
        try:
            os.makedirs(root, exist_ok=True)
        except Exception as e:
            self.show_error("Invalid Save Folder", f"Could not create/access:\n{root}\n\n{e}")
            return

        # Persist as default for next launches
        settings = load_user_settings()
        settings["last_save_root"] = root
        save_user_settings(settings)

        self.cfg.out_dir_root = root

        # read cfg from UI
        self.cfg.fs_req = float(self.fs_box.value())
        s = float(self.scale_box.value())
        self.cfg.scale_nt_per_v = np.array([s, s, s], dtype=float)

        self.cfg.ch_x = self.chx_combo.currentText()
        self.cfg.ch_y = self.chy_combo.currentText()
        self.cfg.ch_z = self.chz_combo.currentText()
        # --- read AUX config from UI ---
        self.cfg.enable_aux = self.aux_enable.isChecked()
        self.cfg.ch_aux1 = self.aux1_combo.currentText()
        self.cfg.ch_aux2 = self.aux2_combo.currentText()
        self.cfg.ch_aux3 = self.aux3_combo.currentText()

        session_dir = core.make_session_dir(self.cfg.out_dir_root)

        try:
            field_ch = [self.cfg.ch_x, self.cfg.ch_y, self.cfg.ch_z]
            aux_ch = []
            if getattr(self.cfg, "enable_aux", False):
                aux_ch = [self.cfg.ch_aux1, self.cfg.ch_aux2, self.cfg.ch_aux3]

            self.ni = core.ContinuousNI(field_ch, self.cfg, session_dir, aux_channels=aux_ch)
            self.ni.start()
        except Exception as e:
            self.show_error("Start failed", str(e))
            self.ni = None
            return

        # Store config snapshot
        import json
        with open(os.path.join(session_dir, "config.json"), "w") as f:
            json_cfg = self.cfg.__dict__.copy()
            json_cfg["scale_nt_per_v"] = [float(x) for x in np.asarray(self.cfg.scale_nt_per_v).reshape(3)]
            json.dump(json_cfg, f, indent=2)

        self.session = core.FieldMapSession(self.cfg, self.ni, session_dir)
        self.session_running = True

        # UI workflow: collapse setup, open measurement + monitor during a run
        self.box_setup.set_collapsed(True)
        self.box_meas.set_collapsed(False)
        self.box_mon.set_collapsed(False)
        
        self.chk_aux.setEnabled(getattr(self.ni, "aux_n", 0) > 0)

        # Toggle UI
        self.btn_start.setText("End Session")
        self.lock_setup(True)
        self.enable_controls(True)

        self.status.setText(f"Session started: {session_dir}")
        self.update_next_label()

    def lock_setup(self, locked: bool):
        # locked=True disables setup controls while running
        for w in [
            self.backend_combo, self.dev_combo,
            self.chx_combo, self.chy_combo, self.chz_combo,
            self.aux_enable, self.aux_dev_combo, self.aux1_combo, self.aux2_combo, self.aux3_combo,
            self.fs_box, self.scale_box,
            self.out_edit, self.btn_refresh
        ]:
            w.setEnabled(not locked)

    def end_session(self):
        if not self.session_running:
            return

        # Attempt export (even if incomplete)
        msg_parts = []
        try:
            ok, msg = core.export_partial_if_possible(self.session)
            msg_parts.append(msg)
        except Exception as e:
            msg_parts.append(f"Export failed: {e}")

        # Stop NI
        try:
            if self.ni is not None:
                self.ni.stop()
        except Exception:
            pass

        # Reset session state
        ended_dir = self.session.session_dir if self.session else "(unknown)"
        self.session = None
        self.ni = None
        self.session_running = False

        # UI back to start mode
        self.btn_start.setText("Start Session")
        self.lock_setup(False)
        self.enable_controls(False)

        # After ending: open setup again, collapse others
        self.box_setup.set_collapsed(False)
        self.box_meas.set_collapsed(True)
        self.box_mon.set_collapsed(True)

        self.status.setText("Session ended. " + " | ".join(msg_parts) + f" | Folder: {ended_dir}")
        self.update_next_label()

    def _draw_grid_preview(self, pt):
        """
        pt = (x,y,z) in index coordinates (1..grid_n). If None: show idle view.

        Visual logic:
        - only the CURRENT z-level is highlighted in blue
        - all other levels are grey
        - current point is orange
        - the dot cloud is rotated 90° in the horizontal plane
        while the black axes/walls view stays unchanged
        """
        ax = self.grid_ax
        ax.clear()

        n = int(self.cfg.grid_n)
        gx, gy, gz = self._grid_coords

        gx_rot = (n + 1) - gy
        gy_rot = gx
        gz_rot = gz

        if pt is not None:
            x_now, y_now, z_now = pt

            # rotate current point with the same mapping
            x_now_rot = (n + 1) - y_now
            y_now_rot = x_now
            z_now_rot = z_now

            # current level only = blue-ish
            m_current = (gz_rot == z_now)
            m_other = ~m_current

            # all non-current planes grey
            ax.scatter(
                gx_rot[m_other], gy_rot[m_other], gz_rot[m_other],
                s=10, alpha=0.18, c="lightgray"
            )

            # current plane highlighted
            ax.scatter(
                gx_rot[m_current], gy_rot[m_current], gz_rot[m_current],
                s=12, alpha=0.45
            )

            # current point in orange
            ax.scatter([x_now_rot], [y_now_rot], [z_now_rot], s=90, marker="o")

            ax.set_title(f"Next: ({x_now},{y_now},{z_now})", fontsize=9)
        else:
            # idle view: all points grey
            ax.scatter(gx_rot, gy_rot, gz_rot, s=10, alpha=0.20, c="lightgray")
            ax.set_title("Next point", fontsize=9)

        # cube framing stays the same
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(0.5, n + 0.5)
        ax.set_zlim(0.5, n + 0.5)

        # hide ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.view_init(elev=22, azim=-55)

        # transparent panes
        try:
            ax.xaxis.pane.set_alpha(0.0)
            ax.yaxis.pane.set_alpha(0.0)
            ax.zaxis.pane.set_alpha(0.0)
        except Exception:
            pass

        self.grid_canvas.draw_idle()


    def update_next_label(self):
        if self.session is None:
            self.lbl_next.setText("Next: (start session)")
            self._draw_grid_preview(None)
            return

        if self.session.cal_mode is not None:
            nxt = self.session.wizard_next_label()
            self.lbl_next.setText(f"Offset {self.session.cal_mode.upper()}: set {nxt} and press 'Capture Step'")
            self._draw_grid_preview(None)  # wizard isn't a map point
            return

        nxt_pt = self.session.next_point()
        if nxt_pt is None:
            self.lbl_next.setText("All points captured. Next: Record END offset, then End Session")
            self._draw_grid_preview(None)
        else:
            self.lbl_next.setText(f"Next point (x,y,z): {nxt_pt}   ({self.session.point_idx}/{len(self.session.points)})")
            self._draw_grid_preview(nxt_pt)

    # ---------------- measurement actions ----------------

    def start_off0(self):
        if self.session is None: return
        self.session.start_wizard("start")
        self.status.setText("START offset wizard started. Follow the instructions.")
        self.update_next_label()

    def start_off1_export(self):
        if self.session is None:
            return
        self.session.start_wizard("end")
        self.status.setText("END offset wizard started. After done, press End Session to export.")
        self.update_next_label()

    def cancel_wizard(self):
        if self.session is None: return
        self.session.cancel_wizard()
        self.status.setText("Wizard cancelled.")
        self.update_next_label()

    def capture_step(self):
        if self.session is None:
            return
        ok, msg = self.session.capture_wizard_step()
        self.status.setText(msg)
        self.update_next_label()

    def record_point(self):
        if self.session is None:
            return
        ok, msg = self.session.record_point()
        self.status.setText(msg)
        self.update_next_label()

    def record_aux_point(self):
        if self.session is None:
            return

        comment, ok_comment = QInputDialog.getText(
            self,
            "AUX Point",
            "Comment / description (optional):"
        )
        if not ok_comment:
            comment = ""

        ok, msg = self.session.record_aux_point(comment.strip())
        self.status.setText(msg)
        self.update_next_label()

    def undo_point(self):
        if self.session is None: return
        ok, msg = self.session.undo_last_point()
        self.status.setText(msg)
        self.update_next_label()

    def retake_point(self):
        if self.session is None: return
        ok, msg = self.session.retake_last_point()
        self.status.setText(msg)
        self.update_next_label()

    # ---------------- monitor toggles ----------------

    def toggle_units(self):
        self.show_nt = self.chk_nt.isChecked()

    def toggle_corr(self):
        self.show_corr = self.chk_corr.isChecked()

    def toggle_pause(self):
        self.paused = self.chk_pause.isChecked()
        
    def toggle_aux(self):
        self.show_aux = self.chk_aux.isChecked()
        self.canvas_aux.setVisible(self.show_aux)

        if self.show_aux:
            # Restore a sensible split when AUX is shown
            self.monitor_split.setSizes([700, 250])

    def clear_plot(self):
        if self.ni is not None:
            self.ni.request_clear()

    def update_monitor(self):
        if self.ni is None or self.session is None:
            return
        if self.paused:
            return

        got = self.ni.get_last_window(self.cfg.plot_window_s)
        if got is None:
            return
        start_idx, data = got  # data in V, shape (3+aux, N) if AUX enabled

        field = data[:3, :]
        aux = data[3:, :] if data.shape[0] > 3 else None

        # optional display correction (START offset only) -> ONLY on field
        if self.show_corr:
            if self.session.off0 is not None:
                off0_vec = np.asarray(self.session.off0["offset_v"], dtype=float).reshape(3)
                field = field - off0_vec[:, None]
            else:
                self.status.setText("Corr view ON but START offset not recorded -> showing RAW")

        # decimate for plotting (same t for both)
        dec = max(1, int(self.cfg.plot_decim))

        field_d = field[:, ::dec]
        n = field_d.shape[1]
        t = (np.arange(n) * dec) / float(self.ni.actual_fs)
        t = t - t[-1] 

        # unit scaling for FIELD only
        if self.show_nt:
            sc = np.asarray(self.cfg.scale_nt_per_v, dtype=float).reshape(3)
            field_plot = (field_d.T * sc).T
            unit_field = "nT"
        else:
            field_plot = field_d
            unit_field = "V"

        mode_field = "corr" if (self.show_corr and self.session.off0 is not None) else "raw"
        self.canvas_field.set_ylabels(unit_field, mode_field)
        self.canvas_field.update_data(t, field_plot)

        # AUX plot: always raw volts, optional visibility
        if self.show_aux and aux is not None and aux.shape[0] > 0:
            aux_d = aux[:, ::dec]
            # If fewer than 3 aux channels are enabled, slice to what's there
            aux_d = aux_d[:3, :]
            self.canvas_aux.set_ylabels("V", "raw")
            self.canvas_aux.update_data(t, aux_d)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

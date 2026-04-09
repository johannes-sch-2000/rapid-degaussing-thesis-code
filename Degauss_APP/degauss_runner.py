#!/usr/bin/env python3
import json
import math
import os
import socket
import sys
import time
from typing import Optional

SCPI_HOST = "127.0.0.1"
SCPI_PORT = 5000
STOP_FLAG = "/tmp/degauss_stop"

def scpi_send(sock: socket.socket, cmd: str) -> None:
    # SCPI server expects newline-terminated commands
    sock.sendall((cmd + "\n").encode("ascii"))

def rp_stop_all(sock: socket.socket) -> None:
    # Best-effort shutdown (both channels)
    scpi_send(sock, "SOUR1:VOLT 0")
    scpi_send(sock, "SOUR2:VOLT 0")
    scpi_send(sock, "OUTPUT1:STATE OFF")
    scpi_send(sock, "OUTPUT2:STATE OFF")

def setup_channel(sock: socket.socket, ch: int, f0: float) -> None:
    scpi_send(sock, f"SOUR{ch}:FUNC SINE")
    scpi_send(sock, f"SOUR{ch}:FREQ:FIX {f0}")
    scpi_send(sock, f"SOUR{ch}:VOLT 0")
    scpi_send(sock, f"SOUR{ch}:VOLT:OFFS 0")
    scpi_send(sock, f"SOUR{ch}:TRIG:SOUR INT")

def build_envelope(params: dict) -> list[float]:
    # Envelope in Vpeak per period
    A_vpp = float(params["amp_vpp"])
    A = A_vpp / 2.0

    Nu = int(params["periods_up"])
    Nh = int(params["periods_hold"])
    Nd = int(params["periods_down"])
    env_type = params.get("envelope", "linear")
    log_decades = float(params.get("log_decades", 3.0))

    if Nu <= 0 or Nh < 0 or Nd <= 0:
        raise ValueError("Periods must satisfy: up>0, hold>=0, down>0")

    env = []

    if env_type == "linear":
        for i in range(1, Nu + 1):
            env.append(A * (i / Nu))
        for _ in range(Nh):
            env.append(A)
        for i in range(1, Nd + 1):
            v = A * (1.0 - (i / Nd))
            env.append(max(0.0, v))
        env[-1] = 0.0

    elif env_type == "log":
        d = max(0.1, log_decades)
        # up: 10^-d -> 10^0
        for i in range(Nu):
            x = i / max(1, Nu - 1)
            up = 10 ** (-d + d * x)
            env.append(A * up)
        for _ in range(Nh):
            env.append(A)
        # down: 10^0 -> 10^-d
        for i in range(Nd):
            x = i / max(1, Nd - 1)
            down = 10 ** (0.0 - d * x)
            env.append(A * down)
        env[-1] = 0.0

    else:
        raise ValueError("envelope must be 'linear' or 'log'")

    return env

def main():
    if len(sys.argv) != 2:
        print("Usage: degauss_runner.py /path/to/run.json", flush=True)
        return 2

    cfg_path = sys.argv[1]
    params = json.loads(open(cfg_path, "r", encoding="utf-8-sig").read())

    out_mode = params.get("out_mode", "OUT1")   # OUT1, OUT2, BOTH
    f0 = float(params["f0_hz"])
    t0 = 1.0 / f0

    env = build_envelope(params)
    N = len(env)

    # Clear any stale stop flag
    try:
        if os.path.exists(STOP_FLAG):
            os.remove(STOP_FLAG)
    except Exception:
        pass

    sock: Optional[socket.socket] = None
    try:
        sock = socket.create_connection((SCPI_HOST, SCPI_PORT), timeout=2.0)

        scpi_send(sock, "GEN:RST")
        rp_stop_all(sock)

        if out_mode == "OUT1":
            setup_channel(sock, 1, f0)
            scpi_send(sock, "OUTPUT1:STATE ON")
        elif out_mode == "OUT2":
            setup_channel(sock, 2, f0)
            scpi_send(sock, "OUTPUT2:STATE ON")
        elif out_mode == "BOTH":
            setup_channel(sock, 1, f0)
            setup_channel(sock, 2, f0)
            scpi_send(sock, "OUTPUT1:STATE ON")
            scpi_send(sock, "OUTPUT2:STATE ON")
        else:
            raise ValueError("out_mode must be OUT1, OUT2, or BOTH")

        # Trigger(s)
        if out_mode in ("OUT1", "BOTH"):
            scpi_send(sock, "SOUR1:TRIG:INT")
        if out_mode in ("OUT2", "BOTH"):
            scpi_send(sock, "SOUR2:TRIG:INT")

        # Handshake for UI timing
        print(f"RUN_START {time.time():.6f}", flush=True)

        start = time.monotonic()

        for i, amp_peak in enumerate(env):
            # Interruptible timing
            t_target = start + i * t0
            while True:
                if os.path.exists(STOP_FLAG):
                    raise KeyboardInterrupt("Stop requested")
                remaining = t_target - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.05, remaining))

            if out_mode == "OUT1":
                scpi_send(sock, f"SOUR1:VOLT {amp_peak:.6f}")
            elif out_mode == "OUT2":
                scpi_send(sock, f"SOUR2:VOLT {amp_peak:.6f}")
            else:  # BOTH
                scpi_send(sock, f"SOUR1:VOLT {amp_peak:.6f}")
                scpi_send(sock, f"SOUR2:VOLT {amp_peak:.6f}")

        print("RUN_DONE", flush=True)
        return 0

    except KeyboardInterrupt:
        print("RUN_STOPPED", flush=True)
        return 1

    except Exception as e:
        print(f"RUN_ERROR {e!r}", flush=True)
        return 3

    finally:
        try:
            if sock is not None:
                rp_stop_all(sock)
        except Exception:
            pass
        try:
            if sock is not None:
                sock.close()
        except Exception:
            pass

if __name__ == "__main__":
    raise SystemExit(main())

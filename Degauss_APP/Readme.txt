Degauss GUI for Red Pitaya (RP-local runner)
===========================================

Overview
--------
This project provides a Windows GUI for running degaussing waveforms on a Red Pitaya.
The GUI itself runs on the PC, but the actual waveform execution is done locally on the
Red Pitaya by `degauss_runner.py`.

This has two major advantages:
1. Better stability during long runs.
2. If the network cable is unplugged during a run, the RP can still finish the waveform
   and ramp back down safely instead of freezing at the last commanded amplitude.

The GUI communicates with the RP using:
- SSH
- SCP

The GUI uploads a small JSON file (`run.json`) to the RP and then starts the runner.

-------------------------------------------------------------------------------
1. Files used in this project
-------------------------------------------------------------------------------

Main PC-side files:
- `degauss_gui.py`
  Main GUI application.
- `degauss_runner.py`
  RP-side runner script. This must be copied to the RP.
- `README.txt`
  This document.

Files/directories created during use:
- `degauss_logs\`
  Local folder on the PC containing JSON run logs.
- `/root/degauss/run.json`
  Current run configuration on the RP.
- `/root/degauss/degauss_runner.py`
  Runner script on the RP.
- `/tmp/degauss_stop`
  Stop flag on the RP. Created to stop a running waveform.

-------------------------------------------------------------------------------
2. What is required on a PC
-------------------------------------------------------------------------------

To RUN the GUI executable on a PC:
- Windows 10 or 11
- OpenSSH Client installed
  (`ssh.exe` and `scp.exe` must be available in PATH)
- Network connection to the Red Pitaya
- SSH key-based login configured for the current Windows user account

To RUN the GUI from source code:
- Python installed on the PC
- A local virtual environment (`.venv`)
- Required Python packages:
  - PySide6
  - pyqtgraph
  - numpy

To REBUILD the executable:
- Everything required to run from source
- PyInstaller

Check whether SSH and SCP are available:
- Open PowerShell and run:
  `where.exe ssh`
  `where.exe scp`

If both commands return paths, OpenSSH Client is installed.

-------------------------------------------------------------------------------
3. What is required on the Red Pitaya
-------------------------------------------------------------------------------

The Red Pitaya must:
- Be reachable over the network
- Allow SSH login as `root`
- Have the SCPI service running
- Have `python3` available
- Have the runner script stored at:
  `/root/degauss/degauss_runner.py`

Recommended RP setup:
- SCPI service enabled and running
- Runner directory:
  `/root/degauss`

Typical service checks on the RP:
- `systemctl is-active redpitaya_scpi`
- `ss -lntp | grep 5000`

-------------------------------------------------------------------------------
4. First-time setup on a new PC
-------------------------------------------------------------------------------

4.1 Install OpenSSH Client
If `where.exe ssh` or `where.exe scp` does not return a path:
- Open Windows Settings
- Go to Apps -> Optional Features
- Install "OpenSSH Client"

4.2 Set up SSH key-based login for the current Windows user
Important:
SSH keys are user-specific. If the GUI is run under another Windows user account,
that user also needs its own SSH key configured.

Open PowerShell and generate a key:
- `ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_ed25519"`

When asked for a passphrase:
- Press Enter for no passphrase, or set one if required by your environment.

Copy the public key to the RP:
- `type "$env:USERPROFILE\.ssh\id_ed25519.pub" | ssh root@<RP_IP> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"`

Test passwordless SSH:
- `ssh -o BatchMode=yes root@<RP_IP> "echo OK"`

Expected result:
- `OK`

If this does not work, do not continue before fixing SSH access.

4.3 Copy the runner to the RP
From the project folder on the PC:
- `scp .\degauss_runner.py root@<RP_IP>:/root/degauss/degauss_runner.py`
- `ssh root@<RP_IP> "chmod +x /root/degauss/degauss_runner.py"`

-------------------------------------------------------------------------------
5. Running the GUI from source
-------------------------------------------------------------------------------

5.1 Create or recreate the local virtual environment
Important:
Do not copy a `.venv` from another PC. Windows virtual environments are not portable.

From the project folder:
- `py -m venv .venv`

Install dependencies:
- `.\.venv\Scripts\python.exe -m pip install --upgrade pip`
- `.\.venv\Scripts\python.exe -m pip install pyside6 pyqtgraph numpy`

5.2 Start the GUI
From the project folder:
- `.\.venv\Scripts\python.exe .\degauss_gui.py`

If PowerShell blocks `.venv\Scripts\Activate.ps1`, do not activate the environment.
Just call `python.exe` inside `.venv\Scripts\` directly as shown above.

-------------------------------------------------------------------------------
6. Running the GUI executable
-------------------------------------------------------------------------------

If an executable build is provided:
- Copy the full `dist\DegaussGUI\` folder (for onedir builds), or
- Copy the single `.exe` file (for onefile builds)

Then launch:
- `DegaussGUI.exe`

Important:
The target PC still needs:
- `ssh.exe`
- `scp.exe`
- SSH keys configured for the current Windows user
- Network access to the RP

-------------------------------------------------------------------------------
7. How to use the GUI
-------------------------------------------------------------------------------

7.1 Main parameters
The GUI provides these main controls:

- RP IP
  IP address of the Red Pitaya

- Output
  Select:
  - OUT1
  - OUT2
  - OUT1+OUT2

- Frequency
  Sine frequency in Hz

- Amplitude
  Entered as Vpp in the GUI

- Ramp-up / Hold / Ramp-down
  Entered as number of periods

- Envelope
  Usually:
  - linear
  - log

- Log folder
  Local folder for JSON run logs

The GUI also shows:
- Preview of the waveform
- Progress marker during the run
- Current status text
- Progress bar

7.2 Start
When Start is pressed, the GUI:
1. Validates settings
2. Creates a local JSON config
3. Uploads it to the RP as `/root/degauss/run.json`
4. Starts the RP runner via SSH
5. Starts a progress timer in the GUI

7.3 Stop
When Stop is pressed, the GUI:
1. Sends a stop request to the RP by creating `/tmp/degauss_stop`
2. The RP runner sees the stop flag
3. The RP ramps the output down safely and disables output

7.4 Closing the GUI
Closing the GUI should also request a stop.
However, always treat stopping as a controlled action:
- Press Stop first if possible
- Then close the GUI

-------------------------------------------------------------------------------
8. Signal timing correction / jump correction
-------------------------------------------------------------------------------

The runner supports update timing correction to minimize small discontinuities when the
amplitude changes from cycle to cycle.

These parameters are passed in `run.json`:
- `update_phase`
- `update_every`

Typical values:
- `"update_phase": 0.93826`
- `"update_every": 1`

Meaning:
- `update_phase` shifts when within the period the amplitude update happens.
- `update_every` defines every how many periods the update is applied.

If the waveform still shows a small jump:
- adjust `update_phase` slightly
- keep `update_every = 1` unless you intentionally want fewer updates

Important:
If the GUI uploads `run.json`, whatever is coded into the GUI becomes the active value.
Manual changes on the RP may be overwritten by the next GUI Start.

-------------------------------------------------------------------------------
9. Logs
-------------------------------------------------------------------------------

After each run, the GUI stores a JSON log locally in the configured log folder.

Typical contents:
- Parameters used
- Start time
- End time
- Success/failure message

This makes each run traceable and reproducible.

-------------------------------------------------------------------------------
10. How to rebuild the executable after code changes
-------------------------------------------------------------------------------

10.1 Go to the project folder
Example:
- `cd "C:\Path\To\degauss_app"`

10.2 Create a clean build environment (recommended)
You may reuse your existing `.venv`, but a dedicated build venv is also fine.

Install build requirements:
- `.\.venv\Scripts\python.exe -m pip install --upgrade pip pyinstaller`
- `.\.venv\Scripts\python.exe -m pip install pyside6 pyqtgraph numpy`

10.3 Clean previous build folders
In PowerShell:
- `Remove-Item -Recurse -Force .\build, .\dist -ErrorAction SilentlyContinue`
- `Remove-Item -Force .\DegaussGUI.spec -ErrorAction SilentlyContinue`

10.4 Build a reliable onedir executable (recommended)
- `.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --windowed --name "DegaussGUI" --collect-submodules pyqtgraph .\degauss_gui.py`

Result:
- `.\dist\DegaussGUI\DegaussGUI.exe`

This is the recommended distribution format for lab use.

10.5 Build a onefile executable (optional)
- `.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --windowed --onefile --name "DegaussGUI" --collect-submodules pyqtgraph .\degauss_gui.py`

Result:
- `.\dist\DegaussGUI.exe`

Note:
Onefile executables:
- start slower
- are more likely to trigger antivirus warnings
- are harder to inspect/debug than onedir builds

-------------------------------------------------------------------------------
11. Recommended release workflow
-------------------------------------------------------------------------------

Before rebuilding:
1. Confirm the GUI works from source
2. Confirm the RP runner on the RP is the current version
3. Confirm Start and Stop work
4. Confirm the waveform looks correct on the scope / NI recording
5. Then rebuild the exe

After rebuilding:
1. Launch the exe on the build PC
2. Test Start/Stop once
3. If correct, copy the resulting build to the target PC

-------------------------------------------------------------------------------
12. Troubleshooting
-------------------------------------------------------------------------------

12.1 SSH timeout in the GUI or exe
Possible causes:
- Wrong RP IP
- Ethernet not connected
- Wrong network adapter
- No SSH key for the current Windows user
- `ssh.exe` available in terminal but not found by the exe environment

Checks:
- `ssh -o BatchMode=yes root@<RP_IP> "echo OK"`
- `ping <RP_IP>`
- `where.exe ssh`
- `where.exe scp`

12.2 GUI opens but Start produces no signal
Checks:
- Confirm `/root/degauss/run.json` updates when Start is pressed
- Confirm the runner starts:
  - `ssh root@<RP_IP> "pgrep -af degauss_runner || true"`
- Confirm the RP can output a basic sine through SCPI

12.3 GUI freezes or becomes unresponsive on older PCs
Mitigations already implemented in the current version:
- SSH/SCP operations moved out of the GUI thread
- Debounced preview updates
- Reduced preview complexity
- More tolerant SSH timeouts

If still needed:
- reduce preview quality further in code
- avoid changing parameters very rapidly
- use the built executable instead of running from source

12.4 Virtual environment does not work on another PC
Cause:
- `.venv` copied from another PC
Fix:
- delete `.venv`
- create a new one locally:
  - `py -m venv .venv`

12.5 VS Code runs the wrong file
If VS Code runs `degauss_runner.py` directly, it will fail unless a JSON file path is supplied.
Normally, users should run:
- `degauss_gui.py`

-------------------------------------------------------------------------------
13. Safety notes
-------------------------------------------------------------------------------

- Always verify the output on a scope after major code changes.
- Test Start/Stop before connecting to the real setup.
- The RP-local runner is safer than streaming amplitude changes from the PC, but it is
  still the user's responsibility to validate behavior before use in the experiment.
- If the GUI is closed or the network drops, behavior depends on the runner and the RP
  state. Always verify that the output returns to zero/off as expected.
- For critical applications, hardware interlocks are still recommended.

-------------------------------------------------------------------------------
14. Recommended folder contents
-------------------------------------------------------------------------------

Recommended project folder layout:

degauss_app\
    degauss_gui.py
    degauss_runner.py
    README.txt
    .venv\
    degauss_logs\
    dist\
    build\

-------------------------------------------------------------------------------
15. Minimal quick-start summary
-------------------------------------------------------------------------------

On a new PC:

1. Install OpenSSH Client
2. Set up SSH key login to the RP
3. Copy `degauss_runner.py` to:
   `/root/degauss/degauss_runner.py`
4. Create local `.venv`
5. Install:
   - PySide6
   - pyqtgraph
   - numpy
6. Run:
   - `.\.venv\Scripts\python.exe .\degauss_gui.py`
7. Verify Start/Stop
8. Rebuild `.exe` with PyInstaller if required

End of README

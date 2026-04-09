# Fieldmap GUI — Build Instructions

## Overview
This guide explains how to build the Fieldmap GUI application into a standalone executable (.exe).

The application typically consists of:
- gui_app.py → graphical user interface
- core.py → data acquisition and processing logic

---

## Requirements

### Python
- Python 3.10 or newer

### Required packages
Install dependencies using pip:

pip install PySide6 numpy matplotlib pyinstaller

Optional:
pip install plotly

---

## Project Structure

Example structure:

fieldmap_app/
│
├── gui_app.py
├── core.py
├── build.spec
├── dist/
├── build/
└── .venv/ (optional)

---

## Build Process

### Step 1 — Open terminal
Navigate to the project directory:

cd <path_to_project>/fieldmap_app

---

### Step 2 — (Optional) Activate virtual environment

Windows:
.venv\Scripts\activate

Linux/macOS:
source .venv/bin/activate

---

### Step 3 — Build executable

Use the provided spec file:

pyinstaller --noconfirm --clean build.spec

---

### Step 4 — Locate executable

The output will be in:

dist/FieldMapDAQ/FieldMapDAQ.exe

---

## Important Notes

### Use the spec file
Always build using:

pyinstaller build.spec

Do NOT build using:

pyinstaller gui_app.py

Reason:
- The spec file defines:
  - application name
  - included files
  - dependencies
  - packaging settings

---

### Clean builds
Use:

--clean

to avoid:
- outdated cached files
- missing updates

---

## Testing the Executable

After building:

1. Launch the application
2. Start a session
3. Verify:
   - live data display
   - field channels (X, Y, Z)
   - AUX channels
   - recording functionality
   - AUX point feature

---

## Development Workflow

1. Modify source code:
   - core.py → logic
   - gui_app.py → interface

2. Test in Python:
   python gui_app.py

3. Rebuild executable:
   pyinstaller --noconfirm --clean build.spec

---

## Common Issues

### Missing modules
Install required package:

pip install <package_name>

---

### GUI does not start
Check:
- PySide6 installation
- Python version compatibility

---

### Wrong executable created
If you see:

dist/gui_app.exe

You built using gui_app.py directly.

Solution:
- delete it
- rebuild using build.spec

---

## Summary

To build the Fieldmap GUI:

1. Navigate to project folder
2. Activate environment (optional)
3. Run:

pyinstaller --noconfirm --clean build.spec

4. Use the executable in:

dist/FieldMapDAQ/

This ensures a stable and fully packaged application.
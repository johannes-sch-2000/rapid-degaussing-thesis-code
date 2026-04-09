# build.spec
# pyinstaller --noconfirm --clean build.spec

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

block_cipher = None

hiddenimports = []
hiddenimports += collect_submodules("pyqtgraph")
hiddenimports += collect_submodules("redpitaya_scpi")

datas = []
# Usually not needed, but if you ever hit "PackageNotFoundError(...)" in frozen mode,
# add copy_metadata("<package>") like you did for nidaqmx stack. :contentReference[oaicite:6]{index=6}
datas += copy_metadata("pyqtgraph")
datas += copy_metadata("redpitaya_scpi")

a = Analysis(
    ["degauss_gui.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DegaussControl",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,                 # set True for debugging builds
    icon="Degauss_icon.ico",       # optional
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="DegaussControl",
)

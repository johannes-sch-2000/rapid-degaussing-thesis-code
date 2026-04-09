# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata

hiddenimports = []
hiddenimports += collect_submodules('matplotlib')
hiddenimports += collect_submodules('plotly')
hiddenimports += collect_submodules('nidaqmx')
hiddenimports += collect_submodules('PySide6')

datas = []
datas += collect_data_files('matplotlib')
datas += collect_data_files('plotly')

datas += copy_metadata("nidaqmx")
datas += copy_metadata("nitypes")
datas += copy_metadata("hightime")

hiddenimports += collect_submodules("nidaqmx")
hiddenimports += collect_submodules("nitypes")
hiddenimports += collect_submodules("hightime")



block_cipher = None

a = Analysis(
    ['gui_app.py'],
    pathex=['.'],
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
    name='FieldMapDAQ',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # set True for debugging
    disable_windowed_traceback=False,
    icon='FieldMapDAQ_icon_minimal.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='FieldMapDAQ',
)

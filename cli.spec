# -*- mode: python -*-

import inspect
import datetime
import torch

block_cipher = None

a = Analysis(['cli.py'],
             pathex=['C:\\Program Files (x86)\\Windows Kits\\10\\Redist\\10.0.18362.0\\ucrt\\DLLs\\x64','C:\\Projects\\NDI_FaceTrack'],
             binaries=[('Processing.NDI.Lib.x64.dll', '.')],
             datas=[('models', 'models'),('styling','styling'),('config.ini','.'),('config.py','.'),('facenet_pytorch','facenet_pytorch')],
             hiddenimports=['pkg_resources.py2_warn','cv2', 'scipy', 'keyboard'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['ptvsd','matplotlib'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='NDI_FaceTrack',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='Tracking')

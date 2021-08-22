# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
excluded_modules = ['torch.distributions']


server = Analysis(['pyinstaller_server.py'],
             pathex=['C:\\Projects\\NDI_FaceTrack'],
             binaries=[('Processing.NDI.Lib.x64.dll', '.')],
             datas=[('models', 'models'),('styling','styling'),('config.ini','.'),('config.py','.')],
             hiddenimports=['cv2', 'keyboard',
                'uvicorn.lifespan.off','uvicorn.lifespan.on','uvicorn.lifespan',
                'uvicorn.protocols.websockets.auto','uvicorn.protocols.websockets.wsproto_impl',
                'uvicorn.protocols.websockets_impl','uvicorn.protocols.http.auto',
                'uvicorn.protocols.http.h11_impl','uvicorn.protocols.http.httptools_impl',
                'uvicorn.protocols.websockets','uvicorn.protocols.http','uvicorn.protocols',
                'uvicorn.loops.auto','uvicorn.loops.asyncio','uvicorn.loops.uvloop','uvicorn.loops',
                'uvicorn.logging'],
             hookspath=None,
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

gui = Analysis(['pyinstaller_gui.py'],
             pathex=['C:\\Projects\\NDI_FaceTrack'],
             binaries=[('Processing.NDI.Lib.x64.dll', '.')],
             datas=[],
             hiddenimports=['cv2', 'keyboard'],
             hookspath=None,
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

MERGE ((server, 'pyinstaller_server', 'NDI_FaceTrack'), (gui, 'pyinstaller_gui', 'Tracking GUI') )

### Server Pyinstaller portion ###
server_pyz = PYZ(server.pure, server.zipped_data,
             cipher=block_cipher)
             
server_exe = EXE(server_pyz,
          server.scripts,
          [],
          exclude_binaries=True,
          name='NDI_FaceTrack',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True)

server_coll = COLLECT(server_exe,
               server.binaries,
               server.zipfiles,
               server.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Tracking Server')

### GUI Pyinstaller ###
gui_pyz = PYZ(gui.pure, gui.zipped_data,
             cipher=block_cipher)
             
gui_exe = EXE(gui_pyz,
          gui.scripts,
          [],
          exclude_binaries=True,
          name='Tracking GUI',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True)

gui_coll = COLLECT(gui_exe,
               gui.binaries,
               gui.zipfiles,
               gui.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Tracking GUI')

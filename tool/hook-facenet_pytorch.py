
from PyInstaller.utils.hooks import get_package_paths


datas = [(get_package_paths('facenet_pytorch')[1],"facenet_pytorch"),]

# NDI_FaceTrack

A Windows desktop application allows Bird Dog PTZ Cameras to automatically track persons of interests, freeing up camera operators. This application uses Machine Learning and CV Techniques to Identify and track both faces and human body figures.


## Quick Start
___
Highly suggest to use a virtual environment when running and install the required modules
```
python -m virtualenv .venv
pip install -r requirements.txt'
```
Launch the FastAPI server. By defaults it runs on 127.0.0.1:8000
Run the Tracking GUI as the direct interface to the program
```
python TrackingServer_FastAPI.py
python TrackingGUI.py
```

## Packaging 
This project supports packaging the program to an .exe using PyInstaller
`cli.spec` is given for easier packaging

## Architecture
Backend: The backend framework is built from a pure [FastAPI](https://github.com/tiangolo/fastapi) framework. The FastAPI handles communication between external applications using websockets and to start namedPipes(FIFO) between the Tracking Module and the GUI. 
Currently it supports data exchanged between:


1) Websockets via Python Pickling
2) Websockets via Qt Framework
3) namedPipes via Python Pickling

Frontend: The GUI interface available in TrackingGUI.py is built on top of the PySide2 (Qt) framework. It communicates with the FastAPI backend using namedPipes(FIFO).

NDI Facetracking uses open multiple open source projects to work seamlessly betwen each other, namely:
- A self-implemented state-of-the-art tracking algorithm based on the following paper: https://www.hindawi.com/journals/mpe/2019/
- A robust and fast facial detection method using MTCNN: https://github.com/ipazc/
- OpenCV Object Trackers
- Custom Tiny Yolov4 running on OpenCV DNN 
- NDI C++  / Python Wrapper Library: https://github.com/buresu/ndi-python


![Alt text](./styling/NDI_Desktop_Application.jpg?raw=tru?raw=true "Title")

https://github.com/Tomas1337/NDI_FaceTrack

An integral part of this project is to have this easily deployable to users in a robust lightweight executable file.
Currently achieves +30FPS on an QuadCore Intel 2.4GHz

# NDI_FaceTrack



A Windows desktop application allows Bird Dog PTZ Cameras to automatically track persons of interests, freeing up camera operators. This application uses Machine Learning and CV Techniques to Identify and track both faces and human body figures.

NDI Facetracking uses open multiple open source projects to work seamlessly betwween each other, namely:
- A self-implemented state-of-the-art tracking algorithm based on the following paper: https://www.hindawi.com/journals/mpe/2019/
- A robust and fast facial detection method using MTCNN: https://github.com/ipazc/
- OpenCV Object Trackers
- Custom Tiny Yolov3 PRN 
- NDI C++  / Python Wrapper Library: https://github.com/buresu/ndi-python
- 
![alt text](https://github.com/Tomas1337/NDI_FaceTrack/styling/NDI_Desktop_Application.jpg?raw=true)

https://github.com/Tomas1337/NDI_FaceTrack


 An integral part of this project is to have this easily deployable to interested in users hence the need for a robust windows executable file.


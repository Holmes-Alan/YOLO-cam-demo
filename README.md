# YOLO-cam-demo
live demo for applying YOLO for object detection

# Live demo on a computer with available webcam
1. create a virtual environment or a local environment with opencv-python and numpy by
```sh
$ pip install opencv-python
$ pip install numpy
```

2. download YOLOv4.weights and put it under "models" folder

wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

3. run the following command to test
```sh
$ python Imagenet_detec.py
```

you can use webcam to test different objects

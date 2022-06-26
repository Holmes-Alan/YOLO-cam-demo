import numpy as np
import cv2
import os
import time


def video_demo():
    weightsPath = "models/yolov4.weights"
    configPath = "models/yolov4.cfg"
    labelsPath = "models/coco.names"


    LABELS = open(labelsPath).read().strip().split("\n")
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    count = cv2.cuda.getCudaEnabledDeviceCount()
    if count > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    capture = cv2.VideoCapture(0)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    fps = 30


    # size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # video_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # videowrite = cv2.VideoWriter('output/MySaveVideo-' + video_time + '.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), 30, size)

    while (True):
        boxes = []
        confidences = []
        classIDs = []
        ref, image = capture.read()
        (H, W) = image.shape[:2]

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 5)
                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                text = "{}".format(LABELS[classIDs[i]])
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        # videowrite.write(image)
        cv2.imshow("Image", image)


        c = cv2.waitKey(1) & 0xFF == ord('q')
        if c:
            capture.release()
            break

video_demo()

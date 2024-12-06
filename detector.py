import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import mediapipe as mp
import tensorflow


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

imgSize = 300
offset = 20
counter = 0

folder = "SignData/A"

labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgCropShape = imgCrop.shape

        aspect_ratio = h / w

        if aspect_ratio > 1:
            k = imgSize / h
            wCalc = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCalc, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCalc) / 2)
            imgWhite[0:imgResizeShape[0], wGap:wCalc+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / h
            hCalc = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (hCalc, imgSize))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCalc) / 2)
            imgWhite[hGap:hCalc + hGap, 0:imgResizeShape[1]] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

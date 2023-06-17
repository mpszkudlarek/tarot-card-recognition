import cv2
from PIL import Image
import imagehash
import os


def getCardsFromFolder(path):
    for name in os.listdir(path):
        image_number = 0
        dirName = path + name + "/"
        os.mkdir("DataSet/" + name)
        for fileName in os.listdir(dirName):
            # Load image, grayscale, Gaussian blur, Otsu's threshold, dilate
            if fileName == "Raw":
                continue
            image = cv2.imread(dirName + fileName)
            original = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            dilate = cv2.dilate(thresh, kernel, iterations=1)

            # Find contours, obtain bounding box coordinates, and extract ROI
            cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            ROI = original[y:y + h, x:x + w]
            cv2.imwrite("DataSet/" + name + "/{}.png".format(image_number), ROI)
            image_number += 1


getCardsFromFolder("TuWpisaćŚcieżkę")


import cv2                                                                      # Importing the opencv
import os

from PIL import Image

import NameFind

#   import the Haar cascades for face and eye ditectionq

face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')

recognise = cv2.face.EigenFaceRecognizer_create(15, 4000)  # creating EIGEN FACE RECOGNISER
recognise.read("Recogniser/trainingDataEigan.xml")                              # Load the training data

# -------------------------     START THE VIDEO FEED ------------------------------------------
cap = cv2.VideoCapture(0)                                                       # Camera object
# cap = cv2.VideoCapture('TestVid.wmv')   # Video object
path = 'dataSet'
ID = 0
imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

for imagePath in imagePaths:
        faceImage = Image.open(imagePath) # Open image and convert to gray
        face = cv2.imread(imagePath)
        faces = face_cascade.detectMultiScale(face, 1.3, 5)
        for (x, y, w ,h) in faces:
            heigth, width = face.size
            if width > 110 & heigth > 110:
                gray_face = cv2.resize((faces[y: y + h, x: x + w]), (110, 110))
            eyes = eye_cascade.detectMultiScale(gray_face)
            for (ex, ey, ew, eh) in eyes:
                ID, conf = recognise.predict(gray_face)  # Determine the ID of the photo
                NAME = NameFind.ID2Name(ID, conf)
                NameFind.DispID(x, y, w, h, NAME, faces)
        cv2.imshow('EigenFace Face Recognition System', face)
        if cv2.waitKey(1) & 0xFF == ord('q'):                                       # Quit if the key is Q
            break
cap.release()
cv2.destroyAllWindows()
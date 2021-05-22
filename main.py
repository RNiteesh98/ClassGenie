from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import pandas as pd
import sys
import os

face_classifier = cv2.CascadeClassifier(
    r'D:\NITEESH\studies\CBIT\4\4.2\Major_Project\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
classifier = load_model(
    r'D:\NITEESH\studies\CBIT\4\4.2\Major_Project\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5')

emotion_labels = ['concentrated', 'Disgust', 'Fear',
                  'intrested', 'calm', 'not_intrested', 'sleepy']

cap = cv2.VideoCapture(0)

i = 0
while os.path.exists(f"file_{i}.txt"):
    i += 1

while True:

    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]

            label = emotion_labels[prediction.argmax()]
            #report = open('exp.txt', 'a')
            #report.write(label + '\n')
            # Determine incremented filename
            file = open(f"file_{i}.txt", "a")
            # ... Do some processing ...
            file.write(label + '\n')

            # print(label)
            #sys.stdout = restorePoint
            label_position = (x, y-10)
            cv2.putText(frame, label, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

file.close()
cap.release()


while os.path.exists(f"outfile_{i}.txt"):
    i += 1
# get file object reference to the file
# file = open("D:\NITEESH\studies\CBIT\4\4.2\Major_Project\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\exp.txt", "r")


sys.stdout = open(f"outfile_{i}.txt", "a")

text = open(f"file_{i}.txt", "r")

# Create an empty dictionary
d = dict()
# Loop through each line of the file
for line in text:
    # Remove the leading spaces and newline character
    line = line.strip()

    # Convert the characters in line to
    # lowercase to avoid case mismatch
    line = line.lower()

    # Split the line into words
    words = line.split(" ")

    # Iterate over each word in line
    for word in words:
        # Check if the word is already in dictionary
        if word in d:
            # Increment count of word by 1
            d[word] = d[word] + 1
        else:
            # Add the word to dictionary with count 1
            d[word] = 1

# Print the contents of dictionary
for key in list(d.keys()):
    print(key, ":", d[key])

cv2.destroyAllWindows()

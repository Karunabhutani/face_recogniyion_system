import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

# Initialize video capture
video = cv2.VideoCapture(0)

# Set the desired camera resolution
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Width
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Height

# Load the Haar cascade classifier for face detection
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load face data and corresponding labels
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

LABELS = LABELS[:FACES.shape[0]]

print('Shape of Faces matrix --> ', FACES.shape)

# Initialize K-nearest neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Initialize background image
imgBackground = cv2.imread("k.png")

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()  # Read frame from video

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]  # Crop face region
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)  # Resize image for prediction
        output = knn.predict(resized_img)  # Predict label
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (0,0,255), -1)
        # Display recognized name
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        attendance = [str(output[0]), str(timestamp)]

    # Resize camera feed and position it in the center
    frame_height, frame_width = frame.shape[:2]
    resized_frame = cv2.resize(frame, (int(frame_width / 2), int(frame_height / 2)))
    frame_center = (imgBackground.shape[1] // 2 - resized_frame.shape[1] // 2,
                    imgBackground.shape[0] // 2 - resized_frame.shape[0] // 2)
    imgBackground[frame_center[1]:frame_center[1] + resized_frame.shape[0],
                  frame_center[0]:frame_center[0] + resized_frame.shape[1]] = resized_frame

    # Display the frame
    cv2.imshow("Frame", imgBackground)

    # Wait for key press
    k = cv2.waitKey(1)
    if k == ord('o'):
        speak("Verified..")
        time.sleep(3)
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
                csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
                csvfile.close()
    if k == ord('q'):
        break

# Release video capture and close all windows
video.release()
cv2.destroyAllWindows()

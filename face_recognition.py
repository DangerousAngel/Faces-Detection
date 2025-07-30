import os
from datetime import date
import cv2 as cv
import face_recognition
import numpy as np
import xlrd
from xlutils.copy import copy as xl_copy
from ultralytics import YOLO # Import YOLO from ultralytics

# Load YOLOv8-face model
yolo_model = YOLO('yolov8n-face.pt') # Load YOLOv8 nano face detection model

# Read current folder path
CurrentFolder = os.getcwd()
image1 = CurrentFolder + '\\arjit.png'
image2 = CurrentFolder + '\\hemant.png'

cam_port = 0
video_capture = cv.VideoCapture(cam_port)

# Load a sample picture and learn how to recognize it.
person1_name = "arjit"
person1_image = face_recognition.load_image_file(image1)
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

person2_name = "hemant"
person2_image = face_recognition.load_image_file(image2)
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]

# Create list of known face encodings and their names
known_face_encodings = [
    person1_face_encoding,
    person2_face_encoding
]
known_face_names = [
    person1_name,
    person2_name
]

# Initialize some variables
face_names = []
process_this_frame = True

# Setting up the attendance Excel file
try:
    rb = xlrd.open_workbook('attendance_excel.xls', formatting_info=True)
except FileNotFoundError:
    print("attendance_excel.xls not found. Please create an empty Excel file named 'attendance_excel.xls'.")
    exit()

wb = xl_copy(rb)
subject_name = input('Please give current subject lecture name: ')
sheet1 = wb.add_sheet(subject_name)
sheet1.write(0, 0, 'Name/Date')
sheet1.write(0, 1, str(date.today()))
row = 1
col = 0
already_attendance_taken = ""

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame of video to 1/4 size for faster processing
    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

    if process_this_frame:
        face_locations = []
        # Perform inference with YOLOv8-face on the RGB small frame
        results = yolo_model(rgb_small_frame, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = box.conf[0]
                if confidence > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # The face_recognition library expects (top, right, bottom, left)
                    # Coordinates are already relative to the `small_frame`
                    top = y1
                    right = x2
                    bottom = y2
                    left = x1
                    face_locations.append((top, right, bottom, left))

        face_encodings = []
        if face_locations: # Only try to encode if faces are detected
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if (already_attendance_taken != name) and (name != "Unknown"):
                sheet1.write(row, col, name)
                col = col + 1
                sheet1.write(row, col, "Present")
                row = row + 1
                col = 0
                print("Attendance taken")
                wb.save('attendance_excel.xls')
                already_attendance_taken = name
            else:
                print("Next student")

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up the face locations for drawing on the original frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv.imshow("Video", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Data save")
        break

video_capture.release()
cv.destroyAllWindows()
import os
from datetime import date
import cv2 as cv
import face_recognition
import numpy as np
import xlrd
from xlutils.copy import copy as xl_copy
from ultralytics import YOLO # Import YOLO from ultralytics
from xlwt import Workbook # Import Workbook for creating new excel file

# Define the directory where known face images are stored
KNOWN_FACES_DIR = "KnownFaces"

# Load YOLOv8-face model
yolo_model = YOLO('yolov8n-face.pt') # Load YOLOv8 nano face detection model

cam_port = 0
video_capture = cv.VideoCapture(cam_port)

# --- Function to load and encode a known face with error handling ---
def load_and_encode_face(image_path, person_name):
    """
    Loads an image, detects and encodes the first face found.
    Handles cases where the image is not found or no face is detected.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path} for {person_name}.")
        return None, None

    try:
        person_image = face_recognition.load_image_file(image_path)
    except Exception as e:
        print(f"Error loading image {image_path} for {person_name}: {e}")
        return None, None

    face_encodings = face_recognition.face_encodings(person_image)
    if len(face_encodings) > 0:
        print(f"Successfully loaded and encoded face for {person_name} from {image_path}")
        return person_name, face_encodings[0]
    else:
        print(f"Warning: No face detected in image {image_path} for {person_name}. Please ensure the image contains a clear face.")
        return None, None

# Load known faces from the specified directory
known_face_encodings = []
known_face_names = []

# Check if the known faces directory exists
if not os.path.exists(KNOWN_FACES_DIR):
    print(f"Error: The '{KNOWN_FACES_DIR}' directory was not found.")
    exit()

# Iterate through files in the known faces directory
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        person_name = os.path.splitext(filename)[0] # Use filename without extension as person's name

        name, encoding = load_and_encode_face(image_path, person_name)
        if encoding is not None:
            known_face_encodings.append(encoding)
            known_face_names.append(name)

# Exit if no known faces were successfully loaded
if not known_face_encodings:
    print(f"No known faces were successfully loaded from the '{KNOWN_FACES_DIR}' directory.")
    print("Ensure the folder contains clear images of faces.")
    exit()

# Initialize some variables
face_names = []
process_this_frame = True

# Setting up the attendance Excel file
try:
    rb = xlrd.open_workbook('Attendance.xls', formatting_info=True)
except FileNotFoundError:
    print("Attendance.xls not found. Creating a new one.")
    # Create a dummy workbook if it doesn't exist
    wb_new = Workbook()
    wb_new.add_sheet('Sheet1') # Add a default sheet
    wb_new.save('Attendance.xls')
    rb = xlrd.open_workbook('Attendance.xls', formatting_info=True) # Re-open the newly created file

wb = xl_copy(rb)
subject_name = input('Subject lecture for the sheet : ')
sheet1 = wb.add_sheet(subject_name)
sheet1.write(0, 0, 'Name/Date')
sheet1.write(0, 1, str(date.today()))
row = 1
col = 0
already_attendance_taken = ""

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame from webcam. Exiting.")
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
        if face_locations: # Only try to encode if faces are detected by YOLOv8
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            # Find the best match among known faces
            if matches: # Only proceed if there's at least one match
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
                print("Attendance taken for", name)
                wb.save('Attendance.xls')
                already_attendance_taken = name
            # else:
                # print("Next student") # This can be very verbose, uncomment if needed for debugging

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up the face locations for drawing on the original frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Set color based on recognition
        if name != "Unknown":
            box_color = (0, 255, 0)  # Green for known faces (BGR)
        else:
            box_color = (0, 0, 255)  # Red for unknown faces (BGR)

        cv.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        cv.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1) # Text color is white

    cv.imshow("Video", frame)

    if cv.waitKey(1) & 0xFF == ord('x'):
        print("Data saved")
        break

video_capture.release()
cv.destroyAllWindows()

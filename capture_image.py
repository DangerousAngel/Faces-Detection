import cv2 as cv
import os

# Capturing the images of students for face recognition
KNOWN_FACES_DIR = "KnownFaces"

# Ensure the known_faces directory exists
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)
    print(f"Created directory: {KNOWN_FACES_DIR}")
    
cam_port = 0
video_capture = cv.VideoCapture(cam_port)

name = input('Enter person name: ')

while True:
    result, image = video_capture.read()
    cv.imshow(name, image)

    # Hit 'q' to quit the window and save the image
    if cv.waitKey(0) & 0xff == ord('x'):
        file_path = os.path.join(KNOWN_FACES_DIR, name + ".png")
        cv.imwrite(file_path, image)
        print("Image Taken")
        break



video_capture.release()
cv.destroyAllWindows()

# **Faces Detection and Attendance System**

This project provides a simple yet effective system for real-time face detection, recognition, and automated attendance tracking using webcam feed. It leverages the power of YOLOv8-face for robust face detection and the face\_recognition library for identifying individuals.

## **Features**

* **Real-time Face Detection:** Utilizes YOLOv8-face to detect faces in live video streams.  
* **Face Recognition:** Identifies known individuals based on pre-registered images.  
* **Automated Attendance:** Records attendance in an Excel file (attendance\_excel.xls) for recognized individuals.  
* **Image Capture Utility:** A separate script to easily capture new images for registering individuals.

## **Requirements**

1. Python 3.x  
2. Install Dependencies:  
   Navigate to the project directory in your terminal and install the necessary libraries using pip:  
   pip install \-r requirements.txt

## **Usage**

### **1\. Capture New Images (capture\_image.py)**

Use this script to easily capture new images of individuals for your recognition database.

python capture\_image.py

* Enter the name of the person when prompted.  
* Press 'q' to capture the image and save it as \[name\].png.  
* Press 'r' to retake the image.

### **2\. Run Face Detection and Attendance (face\_recognition.py)**
   python face\_recognition.py
   
Developed by [DangerousAngel](https://linktr.ee/DangerousAngel)





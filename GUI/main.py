import streamlit as st
import cv2
import imutils
from torch import hub
import time
from threading import Thread
import sys
import platform

# Check if the platform is Windows before importing winsound
if platform.system() == "Windows":
    import winsound

# Motion detection code
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
model = hub.load('ultralytics/yolov5', 'yolov5s')
obj_detection_cap = None
motion_detection_mode = False

def motion_detection(cap):
    if not cap.isOpened():
        st.write("Error: Failed to open camera.")
        return False

    while True:
        _, frame = cap.read()
        if frame is None:
            st.write("Error: Failed to capture frame.")
            break

        fg_mask = bg_subtractor.apply(frame)
        num_white_pixels = cv2.countNonZero(fg_mask)
        motion_threshold = 50000  

        if num_white_pixels > motion_threshold:
            return True
        else:
            return False

def perform_object_detection():
    global obj_detection_cap, motion_detection_mode, model
    while True:
        time.sleep(1)
        
        if motion_detection_mode:
            motion_detected = motion_detection(obj_detection_cap)
            if motion_detected:
                motion_detection_mode = False
                st.write("Motion detected. Switching to object detection mode.")
                obj_detection_cap.release()
                obj_detection_cap = cv2.VideoCapture(0)
            else:
                st.write("No motion detected. LAMP = OFF , WAITING FOR MOTION......")
                continue
        
        ret, img = obj_detection_cap.read()

        if not motion_detection_mode:
            result = model(img, size=640)
            persons = [obj for obj in result.xyxy[0] if obj[5] == 0]

            for person in persons:
                x1, y1, x2, y2, _, _ = person
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)

            if len(persons) > 0:
                # Beep only if winsound is available (i.e., on Windows)
                if 'winsound' in sys.modules:
                    winsound.Beep(1000, 200)
                st.write("Person detected. LAMP = ON ")
            else:
                obj_detection_cap.release()
                motion_detection_mode = True
                st.write("No person detected. Switching back to motion detection mode.")
                obj_detection_cap = cv2.VideoCapture(0)

        st.image(img, channels="BGR")

def start_motion_detection():
    global obj_detection_cap, motion_detection_mode
    obj_detection_cap = cv2.VideoCapture(0)
    motion_detection_mode = True

def stop_motion_detection():
    global obj_detection_cap, motion_detection_mode
    if obj_detection_cap is not None:
        obj_detection_cap.release()
    motion_detection_mode = False

def read_model():
    global model
    model = hub.load('ultralytics/yolov5', 'yolov5s')
    if model:
        st.write("Ready model")
    else:
        st.write("No model")

st.title("SMART MOTION DETECTION | ARTIFICIAL INTELLIGENCE")

st.write("## Real Time Interference")

start_button = st.button("Start.png")
if start_button:
    start_motion_detection()
    t = Thread(target=perform_object_detection)
    t.start()
    st.write("Start Video")

stop_button = st.button("Stop.png")
if stop_button:
    stop_motion_detection()
    if obj_detection_cap:
        obj_detection_cap.release()
    st.write("Stop Video")

read_button = st.button("Read.png")
if read_button:
    read_model()

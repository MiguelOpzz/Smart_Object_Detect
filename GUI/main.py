#LIBRARY IMPORT
from tkinter import * #GUI
from PIL import Image, ImageTk #GUI Untuk logo button
import cv2 # opencv pengelola gambar
import imutils #support opencv memanipulasi gambar
from torch import hub
import winsound #sound
from threading import Thread #biar bisa multitasking
import time #delay 2 program biar tidak mulai barengan atau tumburan

#motion detection code
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

#model
model = hub.load('ultralytics/yolov5', 'yolov5s')

obj_detection_cap = None

motion_detection_mode = False

def motion_detection(cap):
    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return False

    while True:
        _, frame = cap.read()
        if frame is None:
            print("Error: Failed to capture frame.")
            break

        fg_mask = bg_subtractor.apply(frame)

        num_white_pixels = cv2.countNonZero(fg_mask)
        # untuk sensifitas motion detection
        motion_threshold = 50000  

        if num_white_pixels > motion_threshold:
            return True
        else:
            return False

# object detection code
def perform_object_detection():
    global obj_detection_cap, motion_detection_mode, model
    while True:
        time.sleep(1)
        
        # pengecekan motion detection aktif/tidak
        if motion_detection_mode:
            motion_detected = motion_detection(obj_detection_cap)
            if motion_detected:
                motion_detection_mode = False
                print("Motion detected. Switching to object detection mode.")
                obj_detection_cap.release()  # Release object detection camera
                obj_detection_cap = cv2.VideoCapture(0)  # Reinitialize for next use
            else:
                print("No motion detected. LAMP = OFF , WAITING FOR MOTION......")
                continue
        
        ret, img = obj_detection_cap.read()

        # object detection code bila motion detection sudah mati
        if not motion_detection_mode:
            result = model(img, size=640)
            persons = [obj for obj in result.xyxy[0] if obj[5] == 0]  # memfilter manusia dan hewan

            # kalau dia manusia bakal ada kotak yang menunjukan
            for person in persons:
                x1, y1, x2, y2, _, _ = person
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)

            if len(persons) > 0:
                winsound.Beep(1000, 200) 
                print("Person detected. LAMP = ON ")

            # perulangan bila tidak terdekteksi orang
            else:
                obj_detection_cap.release() 
                motion_detection_mode = True
                print("No person detected. Switching back to motion detection mode.")
                obj_detection_cap = cv2.VideoCapture(0)

        # Display the image with detections
        cv2.imshow('Detect', img)

        # emergency exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

# Release resources
cv2.destroyAllWindows()
if obj_detection_cap:
    obj_detection_cap.release()

# Tombol GUI 
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
        print("Ready model")
    else:
        print("No model")


def start():
    start_motion_detection()
    t = Thread(target=perform_object_detection)
    t.start()
    print('Start Video')

def stop():
    stop_motion_detection()
    if obj_detection_cap:
        obj_detection_cap.release()
    print('Stop Video')


def read():
    read_model()

# Main screen
screen = Tk()
screen.title("GUI | SMART MOTION DETECTION | ARTIFICIAL INTELLIGENCE")
screen.geometry("1280x720") # Set the window size

# Background
imagenF = ImageTk.PhotoImage(file="GUI/background.png")

background = Label(image=imagenF, text="Background")
background.place(x=0, y=0, relwidth=1, relheight=1)

# Interface
text1 = Label(screen, text="ARTIFICIAL INTELLIGENCE")
text1.place(x=580, y=10)

text2 = Label(screen, text="REAL TIME INTERFERENCE:")
text2.place(x=570, y=100)

text5 = Label(screen, text=" ")
text5.place(x=110, y=145)

# Buttons
# Start Video
imageSt = ImageTk.PhotoImage(file="GUI/start.png")
start_button = Button(screen, text="Start", image=imageSt, height="40", width="200", bg="green", command=start)
start_button.place(x=100, y=580)

# Stop Video
imageSp = ImageTk.PhotoImage(file="GUI/stop.png")
stop_button = Button(screen, text="Stop", image=imageSp, height="40", width="200", bg="red", command=stop)
stop_button.place(x=980, y=580)

# Read Model
imageRd = ImageTk.PhotoImage(file="GUI/read.png")
read_button = Button(screen, text="Read", image=imageRd, height="40", width="200", bg="blue", command=read)
read_button.place(x=100, y=350)

screen.mainloop()

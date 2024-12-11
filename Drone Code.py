import cv2
import math
import time
import cvzone
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
from djitellopy import tello

# Initialize DJI Tello
me = tello.Tello()
me.connect()
me.streamon()

cap = cv2.VideoCapture(0) # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

# Initialize YOLO model
model = YOLO("../pythonProject/best50COMB.pt")
classNames = ["person"]

prev_frame_time = 0
new_frame_time = 0

# Initialize Tkinter window
root = tk.Tk()
root.title("DJI Tello GUI")

# Create labels for displaying information
info_frame = tk.Frame(root, width=300, height=400, bg="white")
info_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Height Display
height_image = tk.PhotoImage(file="height.png")
height_image_label = tk.Label(info_frame, image=height_image, bg="white")
height_image_label.grid(row=0, column=2, padx=(10, 0))

height_label = tk.Label(info_frame, text="Height: ", font=("Helvetica", 18))
height_label.grid(row=0, column=0, padx=(10, 0))
height_value = tk.Label(info_frame, text="", font=("Helvetica", 18))
height_value.grid(row=0, column=1, padx=(10, 0))

# ToF Display
ToF_image = tk.PhotoImage(file="ToF.png")
ToF_image_label = tk.Label(info_frame, image=ToF_image, bg="white")
ToF_image_label.grid(row=1, column=2, padx=(10, 0))

ToF_label = tk.Label(info_frame, text="ToF: ", font=("Helvetica", 18))
ToF_label.grid(row=1, column=0, padx=(10, 0))
ToF_value = tk.Label(info_frame, text="", font=("Helvetica", 18))
ToF_value.grid(row=1, column=1, padx=(10, 0))

# Battery Display
battery_label = tk.Label(info_frame, text="Battery: ", font=("Helvetica", 18))
battery_label.grid(row=2, column=0, padx=(10, 0))
battery_value = tk.Label(info_frame, text="", font=("Helvetica", 18))
battery_value.grid(row=2, column=1, padx=(10, 0))

battery_image = tk.PhotoImage(file="battery.png")
battery_image_label = tk.Label(info_frame, image=battery_image, bg="white")
battery_image_label.grid(row=2, column=2, padx=(10, 0))

# Person Display
person_image = tk.PhotoImage(file="person.png")
person_image_label = tk.Label(info_frame, image=person_image, bg="white")
person_image_label.grid(row=3, column=2, padx=(10, 0))

person_counter_label = tk.Label(info_frame, text="Person Detector: ", font=("Helvetica", 18))
person_counter_label.grid(row=3, column=0, padx=(10, 0))
person_counter_value = tk.Label(info_frame, text="", font=("Helvetica", 18))
person_counter_value.grid(row=3, column=1, padx=(10, 0))

#Speed Display
speed_image = tk.PhotoImage(file="speed.png")
speed_image_label = tk.Label(info_frame, image=speed_image, bg="white")
speed_image_label.grid(row=4, column=2, padx=(10, 0))

speed_label = tk.Label(info_frame, text="Speed: ", font=("Helvetica", 18))
speed_label.grid(row=4, column=0, padx=(10, 0))
speed_value = tk.Label(info_frame, text="", font=("Helvetica", 18))
speed_value.grid(row=4, column=1, padx=(10, 0))

# Initialize variables for FPS calculation
prev_frame_time = 0

# Function to update information displayed on labels
def update_labels():
    global prev_frame_time, new_frame_time

    height = me.get_height() / 100
    battery = me.get_battery()
    tof = me.get_distance_tof() / 100

    #initial position
    speed_x = me.get_speed_x()
    speed_y = me.get_speed_y()
    speed_z = me.get_speed_z()

    # Calculate the magnitude of the speed vector
    speed_magnitude = math.sqrt(speed_x ** 2 + speed_y ** 2 + speed_z ** 2)

    height_value.config(text=f"{height:.2f} m")
    ToF_value.config(text=f"{tof:.2f} m")
    battery_value.config(text=f"{battery}%")
    speed_value.config(text=f"{speed_magnitude:.2f} m/s")
    root.after(1000, update_labels)  # Update every second

# Function to update video stream
def update_stream():
    global prev_frame_time
    # Read frame from DJI Tello
    success, img = cap.read()
    img = me.get_frame_read().frame
    # Draw bounding boxes
    results = model(img, stream=True)
    bounding_box_count = 0
    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)

            # Class Name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1.5, thickness=2)
            bounding_box_count += 1

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Convert image to RGB format
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert image to PIL format
    pil_img = Image.fromarray(img)

    # Create Tkinter-compatible photo image
    tk_img = ImageTk.PhotoImage(image=pil_img)

    # Update video stream
    video_stream_label.config(image=tk_img)
    video_stream_label.image = tk_img

    # Update person counter label
    person_counter_value.config(text=f"{bounding_box_count}")

    # Print FPS
    print(fps)

    # Update Video stream after 10 milliseconds
    video_stream_label.after(10, update_stream)  # Update every 10 milliseconds

# Create label for video stream
video_stream_label = tk.Label(root, width=1280, height=720)
video_stream_label.pack(side=tk.LEFT)

# Start updating labels and video stream
update_labels()
update_stream()

root.mainloop()

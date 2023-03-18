import cv2
import numpy as np
import time
from PIL import Image, ImageTk
import tkinter as tk


# video input
cap = cv2.VideoCapture('video.mp4')

# trafffic light timeinggg
RED_TIME = 10
YELLOW_TIME = 3
GREEN_TIME = 15

# threshold for car detection
car_threshold = 100

# initial traffic light state
traffic_light_state = "red"

# Define  timerrrrrrrrr
start_time = time.time()

# evnt handler for  keys detections


def keyboard_event_handler(event):
    global traffic_light_state, start_time
    if event == ord('r'):
        traffic_light_state = "red"
        start_time = time.time()
    elif event == ord('y'):
        traffic_light_state = "yellow"
        start_time = time.time()
    elif event == ord('g'):
        traffic_light_state = "green"
        start_time = time.time()


# Create tkinter window and label
root = tk.Tk()
label = tk.Label(root)
label.pack()

# Main loop
def update_frame():
    global traffic_light_state, start_time
    # Capture video frame
    ret, frame = cap.read()
    if not ret:
        return

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                            minLineLength=100, maxLineGap=10)

    # Count the number of cars in the frame
    car_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) > car_threshold:
                car_count += 1

    # Display car count on frame
    cv2.putText(frame, f"Car count: {car_count}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Adjust traffic light timing based on car density
    elapsed_time = time.time() - start_time
    if car_count < 5:
        traffic_light_state = "green"
        start_time = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Start video from beginning
    elif traffic_light_state == "red" and elapsed_time > RED_TIME:
        traffic_light_state = "green"
        start_time = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Start video from beginning
    elif traffic_light_state == "green" and elapsed_time > GREEN_TIME:
        traffic_light_state = "yellow"
        start_time = time.time()
    elif traffic_light_state == "yellow" and elapsed_time > YELLOW_TIME:
        traffic_light_state = "red"
        start_time = time.time()
    if traffic_light_state == "red":
        cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
        cv2.waitKey(1000)  # Pause for 1 second
    elif traffic_light_state == "yellow":
        cv2.circle(frame, (50, 50), 20, (0, 255, 255), -1)
    elif traffic_light_state == "green":
        cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)

    # Convert the frame to RGB format for displaying in tkinter
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to ImageTk format for displaying in tkinter
    image = Image.fromarray(frame)
    image_tk = ImageTk.PhotoImage(image)

    # Update the label with the new image
    label.config(image=image_tk)
    label.image = image_tk
# Handle keyboard events
    key = cv2.waitKey(1)
    if key == ord('q'):
        root.quit()
    elif key == ord('r'):
        traffic_light_state = "red"
        start_time = time.time()
    elif key == ord('y'):
        traffic_light_state = "yellow"
        start_time = time.time()
    elif key == ord('g'):
        traffic_light_state = "green"
        start_time = time.time()

    # Update the tkinter window
    root.update()
    cap.release()
    cv2.destroyAllWindows()
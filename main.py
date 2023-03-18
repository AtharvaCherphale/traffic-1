import cv2
import numpy as np
import time


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

# Define  timerrrrrrrrrr
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


# Main loop
while True:
    # Capture video frame
    ret, frame = cap.read()
    if not ret:
        break

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

    # Draw traffic light on frame
    if traffic_light_state == "red":
        cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
        cv2.waitKey(1000)  # Pause for 1 second
    elif traffic_light_state == "yellow":
        cv2.circle(frame, (50, 50), 20, (0, 255, 255), -1)
    elif traffic_light_state == "green":
        cv2.circle(frame,    (50, 50), 20, (0, 255, 0), -1)
# Display the frame
    cv2.imshow('Traffic Management System', frame)

    # Handle keyboard events
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        traffic_light_state = "red"
        start_time = time.time()
    elif key == ord('y'):
        traffic_light_state = "yellow"
        start_time = time.time()
    elif key == ord('g'):
        traffic_light_state = "green"
        start_time = time.time()


# Release the video source and close all windows
cap.release()
cv2.destroyAllWindows()

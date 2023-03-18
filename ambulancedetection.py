import cv2
import numpy as np
import time

# Set video input source
cap = cv2.VideoCapture('video.mp4')

# Set traffic light timings
RED_TIME = 10
YELLOW_TIME = 3
GREEN_TIME = 15

# Define threshold for car detection
car_threshold = 100

# Load YOLOv4 model
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Define initial traffic light state
traffic_light_state = "red"

# Define initial timer
start_time = time.time()

# Define keyboard event handler
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

    # Detect ambulances using YOLOv4
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:  # 0 corresponds to ambulance in the COCO dataset
                # Trigger warning message and turn traffic light green
                print("Ambulance detected!")
                traffic_light_state = "green"
                start_time = time.time()

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
    
    
        # Apply Haar cascade classifier to detect ambulance
    ambulance_cascade = cv2.CascadeClassifier('ambulance_cascade.xml')
    ambulance_rects = ambulance_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Check if ambulance is detected
    if len(ambulance_rects) > 0:
        # Display message and turn traffic light green
        print("Ambulance detected!")
        traffic_light_state = "green"
        start_time = time.time()

    # Draw traffic light on frame
    if traffic_light_state == "red":
        cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
        cv2.waitKey(1000)  # Pause for 1 second
    elif traffic_light_state == "yellow":
        cv2.circle(frame, (50, 50), 20, (0, 255, 255), -1)
    elif traffic_light_state == "green":
        cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)

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

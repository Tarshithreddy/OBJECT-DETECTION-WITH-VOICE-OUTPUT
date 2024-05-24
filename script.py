import cv2
import numpy as np
import pyttsx3

# Load YOLOv4 object detector
net_v4 = cv2.dnn.readNet("yolo-coco/yolov4-tiny.weights", "yolo-coco/yolov4-tiny.cfg")
layer_names_v4 = net_v4.getLayerNames()
output_layers_v4 = [layer_names_v4[i - 1] for i in net_v4.getUnconnectedOutLayers()]

# Load YOLOv7 object detector
net_v7 = cv2.dnn.readNet("yolo-coco/yolov7-tiny.weights", "yolo-coco/yolov7-tiny.cfg")
layer_names_v7 = net_v7.getLayerNames()
output_layers_v7 = [layer_names_v7[i - 1] for i in net_v7.getUnconnectedOutLayers()]

# Load COCO class labels
labels = open("yolo-coco/coco.names").read().strip().split("\n")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to determine object position
def get_position(x, y, w, h, frame_width, frame_height):
    center_x = x + w // 2
    center_y = y + h // 2

    if center_x <= frame_width // 3:
        W_pos = "left "
    elif center_x <= (frame_width // 3 * 2):
        W_pos = "center "
    else:
        W_pos = "right "

    if center_y <= frame_height // 3:
        H_pos = "top "
    elif center_y <= (frame_height // 3 * 2):
        H_pos = "mid "
    else:
        H_pos = "bottom "

    return H_pos + W_pos

# Function to calculate distance from object dimensions
def calculate_distance(width, height):
    # Assuming focal length and real height of object are known
    # Adjust these parameters according to your camera and setup
    focal_length = 100  # example focal length in pixels
    real_height = 50    # example real height of the object in centimeters

    # Calculate distance using similar triangles
    distance = (real_height * focal_length) / height

    return distance

# Initialize video stream
vs = cv2.VideoCapture(0)

# Set video frame dimensions
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

try:
    while True:  # Infinite loop
        try:
            ret, frame = vs.read()
            if not ret:
                break

            # Get frame dimensions
            (H, W) = frame.shape[:2]

            # Perform object detection with YOLOv4
            blob_v4 = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net_v4.setInput(blob_v4)
            layer_outputs_v4 = net_v4.forward(output_layers_v4)

            # Perform object detection with YOLOv7
            blob_v7 = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net_v7.setInput(blob_v7)
            layer_outputs_v7 = net_v7.forward(output_layers_v7)

            # Initialize lists for detections
            boxes = []
            confidences = []
            class_ids = []

            # Loop over each of the layer outputs for YOLOv4
            for output in layer_outputs_v4:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.4:  # Adjust confidence threshold
                        box = detection[0:4] * np.array([W, H, W, H])
                        (x, y, width, height) = box.astype("int")
                        x = int(x - (width / 2))
                        y = int(y - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Loop over each of the layer outputs for YOLOv7
            for output in layer_outputs_v7:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.4:  # Adjust confidence threshold
                        box = detection[0:4] * np.array([W, H, W, H])
                        (x, y, width, height) = box.astype("int")
                        x = int(x - (width / 2))
                        y = int(y - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maxima suppression (NMS)
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Ensure at least one detection exists
            if len(idxs) > 0:
                # Initialize lists for object descriptions
                descriptions = []

                # Draw bounding boxes and text labels
                for i in idxs.flatten():
                    (x, y, w, h) = boxes[i]
                    color = (0, 255, 0)  # Green color
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # Extract object label and confidence
                    label = str(labels[class_ids[i]])
                    confidence = confidences[i]
                    text = "{}: {:.4f}".format(label, confidence)
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Determine object position
                    position = get_position(x, y, w, h, W, H)

                    # Calculate distance
                    distance = calculate_distance(w, h)

                    # Add description to list
                    descriptions.append("{} at {:.0f}cm {}".format(label, distance, position))

                # Convert descriptions to a single string
                description = ', '.join(descriptions)

                # Print description and speak it
                print(description)
                engine.say(description)
                engine.runAndWait()

            # Display the frame with bounding boxes
            cv2.imshow("Frame", frame)

            # Check for 'q' key press to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Introduce a small delay to prevent high CPU usage
            cv2.waitKey(25)

        except KeyboardInterrupt:
            break

finally:
    # Release the video stream and close windows
    vs.release()
    cv2.destroyAllWindows()

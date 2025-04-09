import cv2
import tensorflow as tf
import numpy as np

# Load the tinted window detector model (TensorFlow)
tinted_model = tf.keras.models.load_model('tinted_window_detector_model.h5')

# Load the YOLOv3 network for car detection
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names for YOLO
layer_names = net.getLayerNames()
# In recent OpenCV versions, net.getUnconnectedOutLayers() returns a 1D array of indices.
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Constants for tinted model input size
IMG_HEIGHT = 224
IMG_WIDTH = 224

def preprocess_roi(roi):
    """
    Resize and normalize the region of interest (ROI) to prepare it for the tinted detection model.
    """
    roi_resized = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT))
    roi_normalized = roi_resized.astype('float32') / 255.0
    roi_expanded = np.expand_dims(roi_normalized, axis=0)
    return roi_expanded

def process_frame(frame):
    """
    Process a single frame:
    - Run YOLOv3 to detect cars.
    - For each car detection, crop the ROI and run tinted window detection.
    - Annotate the frame with detection results.
    """
    height, width = frame.shape[:2]

    # Create blob from image and perform forward pass with YOLOv3.
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    conf_threshold = 0.5  # Confidence threshold for YOLO detections.
    nms_threshold = 0.4   # Non-Maximum Suppression threshold.

    boxes = []
    confidences = []
    class_ids = []

    # Process YOLO detections.
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and classes[class_id] == "car":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove overlapping detections.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Flatten indices if necessary.
    if len(indices) > 0:
        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        elif isinstance(indices, list) and isinstance(indices[0], (list, tuple, np.ndarray)):
            indices = [i[0] for i in indices]

    # Process each detected car and analyze for tinted windows.
    for i in indices:
        x, y, w, h = boxes[i]
        # Draw bounding box for the detected car.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Ensure ROI is within image boundaries.
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, width)
        y2 = min(y + h, height)
        car_roi = frame[y1:y2, x1:x2]
        if car_roi.size == 0:
            continue

        # Preprocess the ROI for the tinted detection model.
        roi_input = preprocess_roi(car_roi)
        tint_prob = tinted_model.predict(roi_input)[0][0]
        tinted = tint_prob > 0.67
        tint_text = "Tinted" if tinted else "Not Tinted"

        # Prepare label with the YOLO detection confidence and tinted window result.
        label = f"Car {confidences[i]:.2f} | {tint_text}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    return frame

def process_image(image_path):
    """
    Load and process an image for car detection and tinted window analysis.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error loading image!")
        return
    result_frame = process_frame(frame)
    cv2.imshow("Detection Result", result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    """
    Open a video file (or webcam) and process frame-by-frame.
    - Resizes each displayed frame to 1920x1080 if the frame is larger than that resolution.
    - Press 'q' to exit the video loop.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = process_frame(frame)

        # Check if the result frame is larger than 1920x1080; if so, resize.
        height, width = result_frame.shape[:2]
        if width > 1920 or height > 1080:
            result_frame = cv2.resize(result_frame, (1920, 1080))

        cv2.imshow("Video Detection", result_frame)

        # Exit on pressing 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mode = input("Process IMAGE or VIDEO? (i/v): ").strip().lower()
    if mode == 'i':
        image_path = input("Enter path to image: ").strip()
        process_image(image_path)
    elif mode == 'v':
        video_path = input("Enter path to video (or leave empty for webcam): ").strip()
        if video_path == "":
            process_video(0)
        else:
            process_video(video_path)
    else:
        print("Invalid option! Please choose either 'i' or 'v'.")

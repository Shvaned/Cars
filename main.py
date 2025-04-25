import os
import cv2
import time
import csv
import random
import string
import numpy as np
import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from pyspark.sql import SparkSession
import pandas as pd
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change for production

# Define folders for uploads and results.
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
RESULT_FOLDER = os.path.join(os.getcwd(), "static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------------------------
# Load models and YOLO configuration
# ---------------------------
tinted_model = tf.keras.models.load_model('tinted_window_detector_model.h5')
colour_model = tf.keras.models.load_model('colours_cnn_model.h5')
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Constants for image sizes
TINT_IMG_HEIGHT, TINT_IMG_WIDTH = 224, 224
COLOUR_IMG_HEIGHT, COLOUR_IMG_WIDTH = 150, 150
SNIPPET_DURATION = 3
COLOUR_CLASSES = ["Black", "White", "Blue", "Green", "Yellow", "Red"]

# ---------------------------
# Helper functions for preprocessing and analysis
# ---------------------------
def preprocess_for_tint(roi):
    roi_resized = cv2.resize(roi, (TINT_IMG_WIDTH, TINT_IMG_HEIGHT))
    roi_normalized = roi_resized.astype('float32') / 255.0
    return np.expand_dims(roi_normalized, axis=0)

def preprocess_for_colour(roi):
    roi_resized = cv2.resize(roi, (COLOUR_IMG_WIDTH, COLOUR_IMG_HEIGHT))
    roi_normalized = roi_resized.astype('float32') / 255.0
    return np.expand_dims(roi_normalized, axis=0)


@app.route('/process_video', methods=["POST"])
def process_video_route():
    # Check that the file exists in the POST request.
    if 'video' not in request.files:
        flash("No video file part")
        return redirect(url_for('upload_page'))
    file = request.files['video']
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for('upload_page'))

    # Secure the filename and save the uploaded video.
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Process the video file.
    cap = cv2.VideoCapture(filepath)
    processed_frames = 0
    tinted_frame_count = 0
    collected_non_tinted = []
    entry_messages = []  # Initialize a list for the entry messages.
    start_time = time.time()
    result_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (time.time() - start_time) >= 10:
            break
        result_frame, is_frame_tinted, non_tinted_colours, _ = process_frame(frame, record_entry_flag=False)
        if is_frame_tinted:
            tinted_frame_count += 1
        if non_tinted_colours:
            collected_non_tinted.extend(non_tinted_colours)
        processed_frames += 1
    cap.release()

    # Calculate tinted percentage.
    if processed_frames > 0:
        tinted_percentage = (tinted_frame_count / processed_frames) * 100
    else:
        tinted_percentage = 0
    final_conclusion = "Tinted" if tinted_percentage > 50 else "Not Tinted"

    # Record a parking entry if final conclusion is Not Tinted and a non-tinted detection exists.
    if final_conclusion == "Not Tinted" and collected_non_tinted:
        # Capture the parking entry message.
        entry_msg = record_entry(collected_non_tinted[0])
        entry_messages.append(entry_msg)

    # Save the last processed frame for display.
    result_filename = "result_video.jpg"
    result_filepath = os.path.join(RESULT_FOLDER, result_filename)
    if result_frame is not None:
        cv2.imwrite(result_filepath, result_frame)

    flash(f"Video processed: {final_conclusion} ({tinted_percentage:.2f}% tinted frames)")
    return render_template("result.html", result_image=result_filename, mode="Video", entry_messages=entry_messages)


@app.route('/process_image', methods=["POST"])
def process_image_route():
    if 'image' not in request.files:
        flash("No file part")
        return redirect(url_for('upload_page'))
    file = request.files['image']
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for('upload_page'))
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    frame = cv2.imread(filepath)
    # Now process_frame returns 4 items.
    result_frame, _, _, entry_messages = process_frame(frame, record_entry_flag=True)
    result_filename = "result_" + filename
    result_filepath = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_filepath, result_frame)

    flash("Image processed successfully!")
    # Pass entry_messages to the result template (it may be an empty list if nothing was recorded)
    return render_template("result.html", result_image=result_filename, mode="Image", entry_messages=entry_messages)

def analyze_car_roi(car_roi):
    roi_height, roi_width = car_roi.shape[:2]
    window_roi = car_roi[0:int(roi_height * 0.6), :]
    roi_input = preprocess_for_tint(window_roi)
    tint_prob = tinted_model.predict(roi_input)[0][0]
    tinted = tint_prob > 0.95
    return tint_prob, tinted

def analyze_car_colour(car_roi):
    roi_input = preprocess_for_colour(car_roi)
    predictions = colour_model.predict(roi_input)[0]
    class_index = np.argmax(predictions)
    return COLOUR_CLASSES[class_index]

def generate_simple_car_id():
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    numbers = str(random.randint(1000, 9999))
    return letters + numbers

@app.route('/upload')
def upload_page():
    return render_template("upload.html")



# ---------------------------
# Parking Allocation: Count cars in a zone image and compute vacancy.
# ---------------------------
def count_cars(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return 0
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416),
                                   swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    conf_threshold = 0.5
    nms_threshold = 0.4
    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    car_count = 0
    if indices is not None and len(indices) > 0:
        # Normalize indices format:
        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        else:
            indices = [i[0] for i in indices]
        for i in indices:
            if classes[class_ids[i]] == "car":
                car_count += 1
    return car_count

def allocate_parking_zone():
    zones = ["A1", "A2", "A3", "A4", "A5"]
    for zone in zones:
        img_path = os.path.join("Zones", f"{zone}.png")
        txt_path = os.path.join("Zones", f"{zone}.txt")
        parked_cars = count_cars(img_path)
        try:
            with open(txt_path, "r") as f:
                capacity = int(f.read().strip())
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")
            continue
        vacancy = capacity - parked_cars
        print(f"Zone {zone}: Capacity = {capacity}, Parked = {parked_cars}, Vacancy = {vacancy}")
        if vacancy > 0:
            new_capacity = capacity - 1
            try:
                with open(txt_path, "w") as f:
                    f.write(str(new_capacity))
                print(f"Allocated parking in zone {zone}. New capacity: {new_capacity}")
                # Here you could also return the full details as a dict.
                return zone
            except Exception as e:
                print(f"Error updating {txt_path}: {e}")
    return None

def record_entry(predicted_colour):
    car_id = generate_simple_car_id()
    allocated_zone = allocate_parking_zone()
    if allocated_zone is None:
        message = "No parking vacancy available. Entry not recorded."
        print(message)
        return message
    stay_duration = round(random.uniform(0.5, 5), 2)
    filename = "Entry_Data.csv"
    write_header = not os.path.exists(filename) or os.stat(filename).st_size == 0
    with open(filename, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["CarID", "Colour", "Parking_Zone", "Stay Duration(Hours)"])
        writer.writerow([car_id, predicted_colour, allocated_zone, stay_duration])
    # Build detailed messages.
    zone_status = f"Zone {allocated_zone}: Parking status updated on server."
    allocation_message = f"Allocated parking in zone {allocated_zone}."
    entry_message = (f"Entry Recorded: CarID: {car_id}, "
                     f"Parking Zone: {allocated_zone}")
    full_message = f"{zone_status}\n{allocation_message}\n{entry_message}"
    print(full_message)
    return full_message

# ---------------------------
# Process frame function (for both image & video processing)
# ---------------------------
def process_frame(frame, record_entry_flag=True):
    height, width = frame.shape[:2]
    non_tinted_colours = []
    entry_messages = []  # if needed, you can accumulate messages here
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416),
                                   swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    conf_threshold = 0.5
    nms_threshold = 0.4
    boxes, confidences, class_ids = [], [], []
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
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if indices is None:
        indices = []  # Set to empty list if no indices returned.
    else:
        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        elif isinstance(indices, list) and isinstance(indices[0], (list, tuple, np.ndarray)):
            indices = [i[0] for i in indices]
    is_frame_tinted = False
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x+w, width), min(y+h, height)
        car_roi = frame[y1:y2, x1:x2]
        if car_roi.size == 0:
            continue
        tint_prob, tinted = analyze_car_roi(car_roi)
        label = f"Car {confidences[i]:.2f} | {'Tinted' if tinted else 'Not Tinted'} ({tint_prob:.2f})"
        if not tinted:
            predicted_colour = analyze_car_colour(car_roi)
            label = f"Allow Car Entry: {'Tinted' if tinted else 'Not Tinted'}"
            if record_entry_flag:
                msg = record_entry(predicted_colour)
                entry_messages.append(msg)
            else:
                non_tinted_colours.append(predicted_colour)
        else:
            is_frame_tinted = True
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame, is_frame_tinted, non_tinted_colours, entry_messages

# ---------------------------
# Dashboard Route: Ribbon with Tabs
# ---------------------------
@app.route('/dashboard')
def dashboard():
    tab = request.args.get('tab', 'entry_data')  # default tab is "Monitor Entry Data"
    content = ""
    if tab == 'entry_data':
        try:
            spark = SparkSession.builder.appName("MonitorData").getOrCreate()
            df = spark.read.csv('Entry_Data.csv', header=True, inferSchema=True)
            pdf = df.toPandas()
            content = pdf.to_html(classes="table table-striped", index=False)
        except Exception as e:
            content = f"<p>Error reading Entry Data: {str(e)}</p>"
    elif tab == 'exit_data':
        try:
            spark = SparkSession.builder.appName("MonitorData").getOrCreate()
            df = spark.read.csv('Exit_Data.csv', header=True, inferSchema=True)
            pdf = df.toPandas()
            content = pdf.to_html(classes="table table-striped", index=False)
        except Exception as e:
            content = f"<p>Error reading Exit Data: {str(e)}</p>"
    elif tab == 'visualize_entry':
        # Use a single image file for entry visualization.
        image_file = os.path.join('static', 'visualize', 'entry', 'Colour.png')
        if not os.path.exists(image_file):
            content = "<p>No entry visualizations available.</p>"
        else:
            img_url = url_for('static', filename='visualize/entry/Colour.png')
            content = f"""
                <div class='row'>
                  <div class='col-md-12 text-center'>
                    <img src='{img_url}' class='img-thumbnail' style='width:1366px; height:768px; object-fit:contain;' alt='Entry Visualization'>
                  </div>
                </div>
            """
    elif tab == 'visualize_exit':
        # Use a single image file for exit visualization.
        image_file = os.path.join('static', 'visualize', 'exit', 'Price.png')
        if not os.path.exists(image_file):
            content = "<p>No exit visualizations available.</p>"
        else:
            img_url = url_for('static', filename='visualize/exit/Price.png')
            content = f"""
                <div class='row'>
                  <div class='col-md-12 text-center'>
                    <img src='{img_url}' class='img-thumbnail' style='width:1366px; height:768px; object-fit:contain;' alt='Exit Visualization'>
                  </div>
                </div>
            """

    return render_template("dashboard.html", active_tab=tab, content=content)

# ---------------------------
# Route to Open Webcam and Process Video
# ---------------------------
@app.route('/open_webcam')
def open_webcam():
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    result_frame = None
    while cap.isOpened() and (time.time() - start_time) < 10:
        ret, frame = cap.read()
        if not ret:
            break
        result_frame, _, _ = process_frame(frame, record_entry_flag=False)
    cap.release()
    result_filepath = os.path.join(RESULT_FOLDER, "webcam_result.jpg")
    if result_frame is not None:
        cv2.imwrite(result_filepath, result_frame)
    flash("Webcam processing complete.")
    return redirect(url_for('dashboard', tab='visualize_entry'))

# ---------------------------
# Static file serving for results (if needed)
# ---------------------------
@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(RESULT_FOLDER, filename)

# ---------------------------
# Home route: redirect to dashboard
# ---------------------------
@app.route('/')
def index():
    return redirect(url_for('dashboard'))

# ---------------------------
# Run Flask App
# ---------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

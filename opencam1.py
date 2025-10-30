from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
from ultralytics import YOLO

app = Flask(__name__)

# Ganti dengan model hasil training kamu
model = YOLO (r"C:\Users\firma\Downloads\Project Akhir\Proyek Akhir\runs\detect\train\weights\best.pt")

# Kamera 1 (laptop cam) dan Kamera 2 (DroidCam)
camera_1 = cv2.VideoCapture(0)
camera_2 = cv2.VideoCapture("")  # contoh URL DroidCam

detected_count = 0
detection_history = []

lock = threading.Lock()

def detect_objects():
    global detected_count, detection_history
    while True:
        ret1, frame1 = camera_1.read()
        ret2, frame2 = camera_2.read()

        if not ret1 and not ret2:
            continue

        # Pilih frame aktif
        frame = frame1 if ret1 else frame2

        # Deteksi dengan YOLO
        results = model(frame, imgsz=640, conf=0.25)
        names = results[0].names
        boxes = results[0].boxes

        count = 0
        for box in boxes:
            cls_id = int(box.cls)
            label = names[cls_id].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Warna berdasarkan class
            if label == "wheelchair_user":
                color = (0, 0, 255)  # merah
                count += 1  # hanya wheelchair_user yang dihitung
            elif label == "empty_wheelchair":
                color = (255, 165, 0)  # oranye
            elif label == "person":
                color = (0, 255, 0)  # hijau
            else:
                color = (255, 255, 255)

            # Gambar bounding box dan label di frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Update data global
        with lock:
            detected_count = count
            if count > 0:
                detection_history.append({
                    "time": time.strftime("%H:%M:%S"),
                    "count": count
                })
                if len(detection_history) > 10:
                    detection_history.pop(0)

        # Encode frame untuk stream
        _, jpeg = cv2.imencode('.jpg', frame)
        yield jpeg.tobytes()

        time.sleep(0.05)

def generate_frames():
    for frame in detect_objects():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/log')
def log_page():
    return render_template('log.html')

@app.route('/settings')
def settings_page():
    return render_template('settings.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

API_KEY = "Delight2025"

@app.route('/data')
def data():
    key = request.args.get("key")
    if key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    with lock:
        return jsonify({
            "count": detected_count,
            "history": detection_history
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

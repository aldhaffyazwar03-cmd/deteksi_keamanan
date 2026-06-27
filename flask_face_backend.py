from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import subprocess
import os
import json
import threading
import time
import base64
import cv2
import numpy as np
import joblib
from threading import Thread
import asyncio
import websockets
import datetime

app = Flask(__name__)
CORS(app)

# Variabel global
detection_active = False
websocket_clients = set()
websocket_server_loop = None  # Event loop untuk WebSocket

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/collect-face-data', methods=['POST'])
def collect_face_data():
    try:
        data = request.get_json()
        nama = data.get('nama')

        if not nama:
            return jsonify({'success': False, 'message': 'Nama harus diisi'})

        print(f"[INFO] Mengumpulkan data wajah untuk: {nama}")

        process = subprocess.Popen(
            ['python', '01_kumpulkan_data.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(input=nama)

        if process.returncode == 0:
            folder_path = os.path.join('data_wajah', nama)
            image_count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])

            return jsonify({
                'success': True,
                'message': f'Data wajah berhasil dikumpulkan',
                'imageCount': image_count,
                'output': stdout
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Error: {stderr}',
                'output': stdout
            })

    except Exception as e:
        print(f"Error in collect_face_data: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/retrain-model', methods=['POST'])
def retrain_model():
    try:
        print("[INFO] Memulai pelatihan ulang model...")

        process = subprocess.Popen(
            ['python', '02_latih_knn.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate()

        if process.returncode == 0:
            return jsonify({'success': True, 'message': 'Model KNN berhasil dilatih', 'output': stdout})
        else:
            return jsonify({'success': False, 'message': f'Error: {stderr}', 'output': stdout})

    except Exception as e:
        print(f"Error in retrain_model: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/start-detection', methods=['POST'])
def start_detection():
    global detection_active

    try:
        if detection_active:
            return jsonify({'success': False, 'message': 'Deteksi sudah berjalan'})

        if not os.path.exists('model_knn.pkl') or not os.path.exists('label_map.npy'):
            return jsonify({'success': False, 'message': 'Model belum dilatih. Latih model terlebih dahulu.'})

        detection_active = True

        detection_thread = Thread(target=run_detection_script)
        detection_thread.daemon = True
        detection_thread.start()

        return jsonify({'success': True, 'message': 'Deteksi real-time dimulai'})

    except Exception as e:
        print(f"Error in start_detection: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

def run_detection_script():
    global detection_active
    try:
        knn = joblib.load('model_knn.pkl')
        label_map = np.load('label_map.npy', allow_pickle=True).item()
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        cam = cv2.VideoCapture(0)
        print("[INFO] Deteksi real-time dimulai...")

        while detection_active:
            ret, frame = cam.read()
            if not ret:
                print("Gagal membaca frame dari kamera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = gray[y:y + h, x:x + w]
                face_img = cv2.resize(roi, (100, 100))
                roi_flat = face_img.flatten()

                pred = knn.predict([roi_flat])
                distances, _ = knn.kneighbors([roi_flat])
                min_dist = distances[0][0]

                if min_dist < 3000:
                    nama = label_map[pred[0]]
                    is_known = True
                else:
                    nama = "TIDAK DIKENAL"
                    is_known = False

                    os.makedirs('log_wajah_tidak_dikenal', exist_ok=True)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"log_wajah_tidak_dikenal/wajah_{timestamp}.jpg"
                    try:
                        cv2.imwrite(filename, face_img)
                        print(f"[INFO] Wajah TIDAK DIKENAL disimpan: {filename}")
                    except Exception as e:
                        print(f"[ERROR] Gagal menyimpan gambar wajah tidak dikenal: {e}")

                    _, buffer = cv2.imencode('.jpg', face_img)
                    face_base64 = base64.b64encode(buffer).decode()

                detection_data = {
                    'type': 'face_detected',
                    'name': nama,
                    'known': is_known,
                    'confidence': float(min_dist),
                    'face_image': face_base64 if not is_known else None
                }

                broadcast_to_websockets(json.dumps(detection_data))

            time.sleep(0.05)

        cam.release()
        print("[INFO] Deteksi real-time dihentikan")

    except Exception as e:
        print(f"Error in detection script: {str(e)}")
        detection_active = False

def broadcast_to_websockets(message):
    global websocket_server_loop
    if websocket_server_loop and websocket_server_loop.is_running():
        async def _send(client, msg):
            try:
                await client.send(msg)
            except websockets.exceptions.ConnectionClosedOK:
                pass
            except Exception as e:
                print(f"Error sending to WebSocket client: {e}")

        for client in list(websocket_clients):
            if client.open:
                asyncio.run_coroutine_threadsafe(_send(client, message), websocket_server_loop)
            else:
                websocket_clients.discard(client)
    else:
        print("[WARNING] WebSocket server loop tidak berjalan.")

# Handler WebSocket
async def websocket_handler(websocket, path):
    websocket_clients.add(websocket)
    print(f"[INFO] WebSocket client connected. Total: {len(websocket_clients)}")

    try:
        await websocket.wait_closed()
    finally:
        websocket_clients.discard(websocket)
        print(f"[INFO] WebSocket client disconnected. Total: {len(websocket_clients)}")

def start_websocket_server():
    global websocket_server_loop
    websocket_server_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(websocket_server_loop)

    start_server = websockets.serve(websocket_handler, "localhost", 8765)

    websocket_server_loop.run_until_complete(start_server)
    print("[INFO] WebSocket server started on ws://localhost:8765")
    websocket_server_loop.run_forever()

if __name__ == '__main__':
    # Jalankan WebSocket di thread terpisah
    ws_thread = Thread(target=start_websocket_server)
    ws_thread.daemon = True
    ws_thread.start()

    print("[INFO] Flask server starting...")
    print("[INFO] Pastikan file berikut ada di direktori yang sama:")
    print("  - 01_kumpulkan_data.py")
    print("  - 02_latih_knn.py")
    print("  - 03_deteksi_peringatan.py")
    print("  - haarcascade_frontalface_default.xml")
    print("\n[INFO] Akses aplikasi di: http://localhost:5000")

    app.run(debug=False, host='0.0.0.0', port=5000)

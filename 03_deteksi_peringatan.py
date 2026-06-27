import cv2
import numpy as np
import joblib

knn = joblib.load('model_knn.pkl')
label_map = np.load('label_map.npy', allow_pickle=True).item()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

print("[INFO] Sistem deteksi berjalan. Tekan 'q' untuk keluar.")
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in wajah:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (100, 100)).flatten()
        pred = knn.predict([roi])
        jarak, tetangga = knn.kneighbors([roi])
        min_dist = jarak[0][0]

        if min_dist < 3000:
            nama = label_map[pred[0]]
            warna = (0, 255, 0)
        else:
            nama = "TIDAK DIKENAL"
            warna = (0, 0, 255)
            print("[PERINGATAN] Orang tidak dikenal terdeteksi!")

        cv2.rectangle(frame, (x, y), (x+w, y+h), warna, 2)
        cv2.putText(frame, nama, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warna, 2)

    cv2.imshow("Sistem Keamanan", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

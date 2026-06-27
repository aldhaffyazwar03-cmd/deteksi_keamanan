import cv2
import os

nama_pengguna = input("Masukkan nama orang: ")
jumlah = 60
folder = "data_wajah"

if not os.path.exists(folder):
    os.makedirs(folder)

path_user = os.path.join(folder, nama_pengguna)
if not os.path.exists(path_user):
    os.makedirs(path_user)

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0

print("[INFO] Mulai ambil gambar wajah. Tekan 'q' untuk keluar.")
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in wajah:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (100, 100))
        cv2.imwrite(f"{path_user}/{count}.jpg", roi)
        count += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Kamera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= jumlah:
        break

cam.release()
cv2.destroyAllWindows()
print(f"[SELESAI] {count} gambar disimpan di folder {path_user}")

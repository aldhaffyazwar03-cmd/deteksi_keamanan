import numpy as np
import os
import cv2
from sklearn.neighbors import KNeighborsClassifier
import joblib

data = []
labels = []
label_map = {}
label_id = 0

folder = "data_wajah"
for nama in os.listdir(folder):
    path = os.path.join(folder, nama)
    if os.path.isdir(path):
        label_map[label_id] = nama
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100)).flatten()
            data.append(img)
            labels.append(label_id)
        label_id += 1

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data, labels)

joblib.dump(knn, 'model_knn.pkl')
np.save('label_map.npy', label_map)

print("[INFO] Model KNN dilatih dan disimpan.")

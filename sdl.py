import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import load_model

# Mendapatkan direktori dari skrip saat ini
base_dir = os.path.dirname(os.path.abspath(__file__))

# Memuat model DNN yang sudah dilatih menggunakan path relatif
model_path = os.path.join(base_dir, 'dnn_model.h5')
model = load_model(model_path)

# Mulai pengambilan video
cap = cv2.VideoCapture(0)  # Ganti dengan 0 jika hanya ada satu webcam
if not cap.isOpened():
    print("Error: Tidak bisa membuka video.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Mendefinisikan label untuk karakter tangan (dari 0 sampai 23 untuk A sampai Y)
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
               12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X',
               23: 'Y'}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Gagal menangkap gambar.")
        continue  # Coba lagi jika gagal menangkap gambar

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Menggambar landmark tangan dan koneksinya
            mp_drawing.draw_landmarks(
                frame,  # gambar untuk menggambar
                hand_landmarks,  # output model
                mp_hands.HAND_CONNECTIONS,  # koneksi tangan
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            data_aux = []
            x_ = []
            y_ = []

            # Mengambil landmark tangan
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Menormalisasi data landmark
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalisasi x
                data_aux.append(y - min(y_))  # Normalisasi y

            # Pastikan data_aux memiliki panjang yang sesuai dengan input model (42)
            if len(data_aux) == 42:
                # Mengubah data menjadi format yang sesuai untuk model
                input_data = np.array([data_aux])  # Membuat input dalam bentuk (1, 42)
                input_data = input_data.astype(np.float32)  # Pastikan tipe data sesuai

                # Melakukan prediksi dengan model
                prediction = model.predict(input_data)
                predicted_class = np.argmax(prediction)  # Mendapatkan kelas dengan probabilitas tertinggi
                predicted_character = labels_dict[predicted_class]  # Mendapatkan label yang sesuai

                # Menentukan koordinat untuk menampilkan teks prediksi
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                # Menggambar kotak dan teks prediksi pada frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Menampilkan frame
    cv2.imshow('frame', frame)

    # Kondisi keluar: tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan capture dan menutup jendela
cap.release()
cv2.destroyAllWindows()

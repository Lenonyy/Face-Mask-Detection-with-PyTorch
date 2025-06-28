
import cv2
import numpy as np
import os
import csv
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from fpdf import FPDF
import pandas as pd
from keras.layers import TFSMLayer
# Model ve sınıflandırıcı yükle
model = load_model("mask_detector.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Klasör ve log dosyası
os.makedirs("veriler", exist_ok=True)
log_path = "veriler/kayit_log.csv"

# CSV log başlığı oluştur
if not os.path.exists(log_path):
    with open(log_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dosya_adi", "tarih_saat", "etiket"])

# Kamera başlat
cap = cv2.VideoCapture(0)
print("Çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera açılamadı.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        resized_face = cv2.resize(rgb_face, (224, 224))
        img_array = img_to_array(resized_face)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        (mask, no_mask) = model.predict(img_array)[0]
        label = "maskeli" if mask > no_mask else "maskesiz"
        color = (0, 255, 0) if label == "maskeli" else (0, 0, 255)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        filename = f"{label}_{timestamp}.jpg"
        full_path = os.path.join("veriler", filename)
        cv2.imwrite(full_path, face)

        # CSV log yaz
        with open(log_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([filename, timestamp, label])

        # Görüntüye çiz
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Maskeli Yüz Algılama (q ile çık)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# PDF RAPOR
df = pd.read_csv(log_path)
toplam_maskeli = df[df['etiket'] == 'maskeli'].shape[0]
toplam_maskesiz = df[df['etiket'] == 'maskesiz'].shape[0]
toplam = toplam_maskeli + toplam_maskesiz
oran_maskeli = (toplam_maskeli / toplam) * 100 if toplam > 0 else 0
oran_maskesiz = (toplam_maskesiz / toplam) * 100 if toplam > 0 else 0

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(200, 10, "Maskeli Yüz Kayıt Raporu", ln=True, align="C")
pdf.set_font("Arial", size=12)
pdf.ln(10)
pdf.cell(200, 10, f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
pdf.ln(10)
pdf.cell(200, 10, f"Toplam Kayıt: {toplam}", ln=True)
pdf.cell(200, 10, f"Maskeli Yüz Sayısı: {toplam_maskeli} (%{oran_maskeli:.1f})", ln=True)
pdf.cell(200, 10, f"Maskesiz Yüz Sayısı: {toplam_maskesiz} (%{oran_maskesiz:.1f})", ln=True)
rapor_adi = f"veriler/rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
pdf.output(rapor_adi)
print(f"PDF rapor oluşturuldu: {rapor_adi}")

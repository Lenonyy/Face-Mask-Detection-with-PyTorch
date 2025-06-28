
📁 MASKELİ YÜZ ALGILAMA MASAÜSTÜ UYGULAMASI (CV2)
=================================================

1. Dosyayı Çalıştırmak İçin:
----------------------------
- Python 3.8 veya üzeri yüklü olmalı
- Gerekli kütüphaneler:
    pip install opencv-python tensorflow fpdf pandas

2. Uygulamayı Başlatmak:
------------------------
    python maskeli_yuz_cv2_gui_loglu_pdfli.py

3. Ne Yapar?
------------
✅ Kameradan görüntü alır  
✅ Yüzleri algılar ve maske durumunu belirler  
✅ Her yüzü 'veriler/' klasörüne kaydeder  
✅ Kayıtları 'kayit_log.csv' dosyasına yazar  
✅ Otomatik PDF rapor üretir

⚠️ Önemli:
---------
Bu dosyayla birlikte aynı klasöre `mask_detector.model` model dosyasını koymalısınız.

İyi çalışmalar!

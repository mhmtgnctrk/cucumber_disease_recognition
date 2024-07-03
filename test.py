import cv2
import numpy as np
from tensorflow.keras.models import load_model

disease_model = load_model('cucumber_disease_model.h5')

def leaf_detection(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Görüntü okunamadı: {image_path}")
            return None
        
        # Burada görüntü işleme işlemleri yapabilirsiniz (ör. kenar tespiti, kontur bulma)
        # Örneğin, Canny kenar tespiti ve kontur bulma:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_leaves = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                detected_leaves.append((x, y, x + w, y + h))
        
        return image, detected_leaves
    
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return None

def detect_leaves_and_check_disease(image_path):
    try:
        image, detected_leaves = leaf_detection(image_path)
        
        if image is None:
            return
        
        for leaf_coords in detected_leaves:
            x1, y1, x2, y2 = leaf_coords
            leaf_image = image[y1:y2, x1:x2]
            
            # Giriş verisini modelin beklediği şekilde düzleştirin veya yeniden şekillendirin
            leaf_image_resized = cv2.resize(leaf_image, (224, 224))
            leaf_image_processed = leaf_image_resized / 255.0  # Normalizasyon
            
            # Giriş verisini modelin beklediği şekilde yeniden şekillendirin veya düzleştirin
            leaf_image_processed = np.expand_dims(leaf_image_processed, axis=0)
            
            # Hastalık tahmini yapın
            prediction = disease_model.predict(leaf_image_processed)
            
            # Tahmin sonuçlarına göre işlem yapın (ör. kırmızı veya yeşil kare çizme)
            if prediction < 0.5:  # Örnek bir sınırlama, gerçek değerlerinize göre ayarlayın
                color = (0, 255, 0)  # Yeşil renk
            else:
                color = (0, 0, 255)  # Kırmızı renk
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        cv2.imshow("Detected Leaves", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Hata oluştu: {e}")

# Test etmek için:
image_path = "data/v3/image_USB_VID_0AC8_PID_C40A_MI_00_8_15623C2F_0_0000_2024-05-22_15-07-17.jpg"  # Test edeceğiniz görüntü yolunu buraya girin
detect_leaves_and_check_disease(image_path)

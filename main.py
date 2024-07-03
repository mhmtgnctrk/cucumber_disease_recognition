import pandas as pd
import numpy as np
import cv2
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import visualkeras

# CSV dosyasını okuma
data = pd.read_csv('data/v3_pot_data.csv')
data['entry_time'] = pd.to_datetime(data['entry_time'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
data = data.dropna(subset=['entry_time'])
print(data.head())

# Resim dosya isimlerinden tarih bilgisi çıkarma
def extract_timestamp(filename):
    date_str = filename.split('_')[-2] + "_" + filename.split('_')[-1].split('.')[0]
    return datetime.strptime(date_str, '%Y-%m-%d_%H-%M-%S')

# Resimlerin yüklenmesi
def load_images_from_folder(folder):
    images = []
    timestamps = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                timestamps.append(extract_timestamp(filename))
    return images, timestamps

images, image_timestamps = load_images_from_folder('data/v3')
print(f'Loaded {len(images)} images.')

# Yüksek pozlama sorunlarını eleme
def is_overexposed(img, threshold=240):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > threshold

filtered_images = []
filtered_timestamps = []
for img, timestamp in zip(images, image_timestamps):
    if not is_overexposed(img):
        filtered_images.append(img)
        filtered_timestamps.append(timestamp)

print(f'Filtered to {len(filtered_images)} images.')

# Resimlerin etiketlenmesi
threshold_date = datetime.strptime('2024-05-17_08-02-05', '%Y-%m-%d_%H-%M-%S')
labels = []
for timestamp in filtered_timestamps:
    if timestamp > threshold_date:
        labels.append(1)  # Hastalıklı
    else:
        labels.append(0)  # Sağlıklı

labels = to_categorical(labels)

# Resimleri yeniden boyutlandırma
def resize_images(images, target_size=(128, 128)):
    resized_images = [cv2.resize(img, target_size) for img in images]
    return resized_images

target_size = (128, 128)
resized_images = resize_images(filtered_images, target_size)

# Eğitim ve test setlerinin oluşturulması
X_train = np.array(resized_images)
y_train = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# CNN Modeli Kurulumu
image_height, image_width, channels = X_train[0].shape
num_classes = len(np.unique(labels))

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, channels)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme ve doğrulama
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Modeli kaydet
model.save('cucumber_disease_model.h5')

visualkeras.layered_view(model, legend=True, show_dimension=True,to_file='output.png').show()

print("Model başarıyla kaydedildi.")
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_data(directory):
    X = []
    y = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            X.append(img_array)
            if directory == 'data/yaprak_var':
                y.append(1)  # yaprak var ise etiket 1
            elif directory == 'data/yaprak_yok':
                y.append(0)  # yaprak yok ise etiket 0
    return X, y

# Verileri yükleyelim
X_leaf, y_leaf = load_data('data/yaprak_var')
X_no_leaf, y_no_leaf = load_data('data/yaprak_yok')

# Verileri numpy dizisine dönüştürelim
X_leaf = np.array(X_leaf)
X_no_leaf = np.array(X_no_leaf)
y_leaf = np.array(y_leaf)
y_no_leaf = np.array(y_no_leaf)

# Veri setini eğitim ve test setlerine ayıralım
X_train, X_test, y_train, y_test = train_test_split(np.concatenate((X_leaf, X_no_leaf), axis=0),
                                                    np.concatenate((y_leaf, y_no_leaf), axis=0),
                                                    test_size=0.2,
                                                    random_state=42)

# Veri seti boyutlarını kontrol edelim
print("Eğitim veri seti boyutu:", X_train.shape)
print("Test veri seti boyutu:", X_test.shape)

# Model oluşturma
model = Sequential()

# Convolutional katmanlar
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

# Düzleştirme ve yoğun katmanlar
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sınıflandırma için sigmoid aktivasyonu

from keras.optimizers import SGD

opt = SGD(learning_rate=0.002, momentum=0.8)

# Modeli derleme
model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Test seti doğruluğunu değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Test seti doğruluğu: {test_acc * 100:.2f}%")

from tensorflow.keras.models import load_model, save_model

# Modeli kaydetme (Keras formatında)
save_model(model, 'leaf_detection_model.keras')
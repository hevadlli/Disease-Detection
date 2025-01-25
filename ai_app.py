import numpy as np
from PIL import Image
import json
import tensorflow as tf
from tensorflow import keras
import os
print(tf.__version__)

# Kelas penyakit
disease_class = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

# Load model pre-trained
model = keras.models.load_model(
    'E:/Code/zona_ai-main/zona_ai-main/model/efficientnetb3-Chicken Disease-98.14.h5', compile=False)
model.trainable = False



def compute_zona_farm_vision_request(image_path):
    # Buka image lokal
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image)

    # Normalisasi nilai pixel jika diperlukan oleh model
    image = image / 255.0  # Normalisasi

    # Tambahkan dimensi batch
    image = np.expand_dims(image, axis=0)

    print("Shape of the image:", image.shape)

    # Prediksi dengan model
    result = model.predict(image)
    # Terapkan softmax untuk probabilitas
    result = tf.nn.softmax(result[0]).numpy()
    print("Prediction results (after softmax):", result)

    # Format output
    output = {
        disease_class[0]: float(result[0]),
        disease_class[1]: float(result[1]),
        disease_class[2]: float(result[2]),
        disease_class[3]: float(result[3])
    }

    return json.dumps(output, indent=4)


# Jalankan fungsi ini dengan path file gambar lokal
if __name__ == '__main__':
    image_path = "E:/Code/zona_ai-main/zona_ai-main/kaggle/input/chicken-disease-1/Train/cocci.12.jpg"

    if os.path.exists(image_path):
        print("Image path: ", image_path)
        prediction = compute_zona_farm_vision_request(image_path)
        print("Prediction output:")
        print(prediction)
    else:
        print("File tidak ditemukan. Harap masukkan path yang valid.")

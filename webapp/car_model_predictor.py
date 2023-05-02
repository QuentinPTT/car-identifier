import numpy as np
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

def predict_car_model(image_path, class_labels, model_path='car_model_classifier_vgg16.h5'):
    model = load_model(model_path)
    
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    class_idx = np.argmax(prediction)

    return class_labels[class_idx], prediction[0][class_idx]

# Load class labels (car model names) from the file
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

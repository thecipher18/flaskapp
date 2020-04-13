import base64
import numpy as np
import io
from PIL import Image
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_model():
    global model 
    model = tf.keras.models.load_model('kerasmodel.h5')
    print("2")
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    print("3")
    if image.mode == "RGB":
        image = image.convert("L")
    print("7")
    image = image.resize(target_size)
    print("8", type(image))
    image = img_to_array(image)
    print("9", image.shape)
    image = np.expand_dims(image, axis = 0)
    image2 = np.expand_dims(image, 1)
    print("10", image.shape," image2: ", image2.shape)
    return image

print(" Loading Keras model...")
get_model()
print("4")
# global graph 
# graph = tf.compat.v1.get_default_graph()
CATEGORIES = ["apple","banana", "bee", "car"]

@app.route('/predict', methods=["POST"])
def predict():
    print("1")
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    print(type(image))
    processed_image = preprocess_image(image, target_size=(28,28))
    print("here")
    # model.predict():
    # with graph.as_default():
    model = tf.keras.models.load_model('kerasmodel.h5')
    prediction = model.predict(processed_image).tolist()
    print("predicted")
    response = {
        'result': CATEGORIES[prediction.index(max(prediction))],
        'prediction': prediction
    }
    return jsonify(response)

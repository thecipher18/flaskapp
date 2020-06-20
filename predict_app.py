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
from flask_mysqldb import MySQL

app = Flask(__name__)

# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = 'root'
# app.config['MYSQL_DB'] = 'MyDB'

mysql = MySQL(app)

def get_model():
    global model 
    model = tf.keras.models.load_model('kerasmodel.h5')
    print("2")
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    print("3")
    if image.mode == "RGB":
        image = image.convert("L")
    print(image)
    image = image.resize(target_size)
    print("image array:", image)
    print("8", type(image))
    image = img_to_array(image)
    print("9", image.shape)
    image = np.expand_dims(image, axis = 0)
    print("image shape after expand dims", image.shape)
    # image2 = np.expand_dims(image, 1)
    # print("10", image.shape," image2: ", image2.shape)
    return image

print(" Loading Keras model...")
get_model()
print("4")
# global graph 
# graph = tf.compat.v1.get_default_graph()

def max_value(inputlist):
    return max([sublist[-1] for sublist in inputlist])

CATEGORIES = ["airplane","apple","axe","banana","baseball","bee","bus","car","diamond",
              "grapes","grass","hand","pineapple","tornado"]


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
        'result': CATEGORIES[prediction[0].index(max(prediction[0]))],
        'prediction': max(prediction[0])
    }
    return jsonify(response)


# @app.route('/hello', methods=['POST'])
# def hello():
#     message = request.get_json(force=True)
#     name = message['name']
#     response = {
#         'greeting': 'Hello' + name
#     }
#     return jsonify(response)

@app.route('/single', methods=["POST"])
def single():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(28,28))
    # model.predict():
    # with graph.as_default():
    model = tf.keras.models.load_model('kerasmodel.h5')
    prediction = model.predict(processed_image).tolist()
    print("predicted")
    print(prediction)

    time = message["time"]
    answer = message["answer"]


    index = CATEGORIES.index(answer)
    modelacc = prediction[0][index]
    myScore = time * modelacc

    response = {
        'modelAnswer': CATEGORIES[prediction[0].index(max(prediction[0]))],
        'score': myScore,
    }
    return jsonify(response)


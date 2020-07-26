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
from cv2 import cv2
import matplotlib.pyplot as plt
import matplotlib
import random
import os

app = Flask(__name__)

mysql = MySQL(app)

def get_model():
    global model 
    model = tf.keras.models.load_model('kerasmodel.h5')
    print(" * Model loaded!")



def deletenp(data, top, right, bot, left):
    data = data
    for i in range(top):
        data = np.delete(data, 0, 0)
        
    data = np.rot90(data)
    for i in range(right):
        data = np.delete(data, 0, 0)
      
    data = np.rot90(data)
    for i in range(bot):
        data = np.delete(data, 0, 0)
        
    data = np.rot90(data)
    for i in range(left):
        data = np.delete(data, 0, 0)
        
    data = np.rot90(data)
    return data

def check255(array):
    for i in array:
            if(i != 255):
                return 1

def crop(data):
    #crop top
    top = 0
    for i in data: 
        if(check255(i)):
            break
        else:
            top += 1
            continue
            
    #crop right
    rotright = np.rot90(data)
    right = 0
    for i in rotright:
        if(check255(i)):
            break
        else:
            right += 1
            continue
            
    #crop bottom
    bot = 0
    rotbot = np.rot90(rotright)
    for i in rotbot:
        if(check255(i)):
            break
        else:
            bot += 1
            continue
            
    #crop left
    left = 0
    rotleft = np.rot90(rotbot)
    for i in rotleft:
        if(check255(i)):
            break
        else:
            left += 1
            continue
    minval =  min(top, right, left, bot)       
    crop_array = deletenp(data, minval, minval, minval, minval)
    return crop_array

def preprocess_image(image):
    IMG_SIZE = 28 
    if image.mode == "RGB":
        image = image.convert("L")
    cv_image = np.array(image)

    message = request.get_json(force=True)
    answer = message["answer"]
    username = message["img_name"]
    name = answer+ str(random.randint(1,999))+ '.jpg'
    dirName = './images/'+answer+'/'
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        plt.imsave(dirName+name, cv_image, format='jpg')
    else:    
        plt.imsave(dirName+name, cv_image, format='jpg')
        
    img_array = crop(cv_image)
    direc = './static/images/'
    step1 = username +"1"+ '.jpg'
    if not os.path.exists(direc):
        os.makedirs(direc)
        plt.imsave(direc+step1, img_array, format='jpg')
    else:    
        plt.imsave(direc+step1, img_array, format='jpg')


    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_AREA)
    step2 = username +"2"+ '.jpg'
    if not os.path.exists(direc):
        os.makedirs(direc)
        plt.imsave(direc+step2, new_array, format='jpg')
    else:    
        plt.imsave(direc+step2, new_array, format='jpg')
    # image = image.resize(target_size)
    # print("image array:", image)
    # print("8", type(image))
    # image = img_to_array(image)
    # print("9", image.shape)
    # image = np.expand_dims(image, axis = 0)
    # print("image shape after expand dims", image.shape)
    # # image2 = np.expand_dims(image, 1)
    # # print("10", image.shape," image2: ", image2.shape)
    # return image
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  

print(" Loading Keras model...")
get_model()
# global graph 
# graph = tf.compat.v1.get_default_graph()

def max_value(inputlist):
    return max([sublist[-1] for sublist in inputlist])

CATEGORIES = ["apple","banana","baseball","bee","camera","car","clock","diamond","hand","tornado"]



@app.route('/predict', methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image)
    # model.predict():
    # with graph.as_default():
    model = tf.keras.models.load_model('kerasmodel.h5')
    prediction = model.predict(processed_image).tolist()

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
    processed_image = preprocess_image(image)
    # model.predict():
    # with graph.as_default():
    model = tf.keras.models.load_model('kerasmodel.h5')
    prediction = model.predict(processed_image).tolist()

    time = message["time"]
    answer = message["answer"]

    index = CATEGORIES.index(answer)
    modelacc = prediction[0][index]
    if 0.95 > modelacc > 0.3:
        if time < 10:
            myScore = int(modelacc*100) + 5
        if 10 < time < 20:
            myScore = int(modelacc*100) + 2
        else:
            myScore = int(modelacc*100)
    else:
        myScore = int(modelacc*100)
    # myScore = int(time / (1-modelacc)) #inverse efficiency score
    if answer not in CATEGORIES:
        myScore = 0
    response = {
        'modelAnswer': CATEGORIES[prediction[0].index(max(prediction[0]))],
        'score': myScore,
    }
    return jsonify(response)


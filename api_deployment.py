#1 - import libraries
import os
import requests
import numpy as np
import tensorflow as tf

from keras.preprocessing.image import save_img,load_img
from flask import Flask,request,jsonify

#2 - import model qnd weights
with open("fashion_model_flask.json","r")as f:
    model_json=f.read()
model=tf.keras.models.model_from_json(model_json)
model.load_weights("fashion_model_flask.h5")

#3- Define the Flask app
app=Flask(__name__)
@app.route("/api/v1/<string:img_name>",methods=["POST"])

#4 - Construct the app
def classify_image(img_name):
    upload_dir = "upload/"
    image=load_img(upload_dir+img_name)
    classes=[" T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker""Bag","Ankle boot"]
    prediction=model.predict([image.reshape(1,28*28)])
    return jsonify({"object_detected":classes[np.argmax(prediction[0]) ]})

#♦5- Deploy on the developement server
app.run(port=5000,debug=False)

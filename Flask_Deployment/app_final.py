from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from sklearn import preprocessing

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model_5k550_with_insofe.h5'
TARGET = pd.read_csv("./models/targets.csv")
TARGET = TARGET.set_index('Encoded_Label')

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    #img = image.load_img(img_path, target_size=(50, 50))
    
    img = cv2.imread(img_path)
    img = cv2.resize(img,(50,50))
    
    # Preprocessing the image
    #x = image.img_to_array(img)
    #x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)
    x = img.reshape(1,50,50,3)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    
    pred2 = np.sort(preds)
    pred2 = np.fliplr(pred2)
    pred2 = pred2.reshape(500,)
    pred2 = pred2[:5]
    print(pred2)

    pred1 = preds.argsort()
    pred1 = np.fliplr(pred1)
    pred1 = pred1.reshape(500,)
    pred1 = pred1[:5]
    print(pred1)

    result = TARGET.loc[pred1]
    result['Percentage Relevant'] = pred2 * 100

    result = result[['Original_Label', 'Percentage Relevant']][result['Percentage Relevant'] > 0]

    #result = str(TARGET['Original_Label'][TARGET['Encoded_Label'].isin(pred1)]) + str('\n') + str(pred2)

    #print(result.to_string(index = False))
    #return result
    
    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        #pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = preds.to_string(index = False, header = False)               # Convert to string
        result = result.replace("\n", ",")
        #result = result.replace("<br />", """ "+"<br />"+" """)
        #result = result.replace("Label", "Label ----->  ")
        result = result.replace("logo", "logo ----->  ")
        #print(result)
        print(result)
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)
    app.debug = True
    app.run(host='0.0.0.0')

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()

import numpy as np
from PIL import Image 
from io import BytesIO
from keras.utils import load_img,img_to_array
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from flask import Flask,redirect, url_for, request, render_template

import flask
app = Flask(__name__)

best = load_model("save_model4.h5")
print('Model Loaded')

#Adding image pre-processing function
def predict_tumor(img_path):
    print("Please Upload Our Proper Image")
    image = Image.open(BytesIO(img_path)) # Resize and preprocess the image
    image = image.resize(size=(224,224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    if best.predict(img)[0][0]>0.45:  # just a threshold
        return "Image indicates Presence of Brain Tumor"
    else:
        return "This is a Healthy Brain"

@app.route('/', methods=['GET'])
def welcome():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        imagefile=request.files['imagefile']
        if imagefile:
            image_path = "./static/" + imagefile.filename
            imagefile.save(image_path)
            return render_template('index.html',prediction=predict_tumor(imagefile),imageloc=imagefile.filename)
    return render_template('index.html',prediction=predict_tumor(imagefile),imageloc = None)

if __name__ == "__main__":
	app.run(port=5000)

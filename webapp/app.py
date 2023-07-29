import os
from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
import cv2

app = Flask(__name__)

model = tf.keras.models.load_model("../model.h5")

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    file = request.files['image']
    if file:
        # Save the uploaded image to a file in the 'uploads' folder
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Read the saved image using cv2.imread
        test_img = cv2.imread(file_path)

        # Perform your image processing and prediction here (the code you already have)

        # For demonstration purposes, let's return the prediction result as a response
        test_input = cv2.resize(test_img, (256, 256))
        test_input = test_input.reshape((1, 256, 256, 3))
        test_input = test_input / 255.0  # Normalize pixel values to [0, 1]

        # Make the prediction
        prediction = model.predict(test_input)

         # Interpret the prediction result
        if prediction[0][0] >= 0.5:
            prediction_result = "The model predicts the image is a dog."
        else:
            prediction_result = "The model predicts the image is a cat."

        # Render the index.html template with the prediction result
        return render_template('index.html', prediction_result=prediction_result)

    return "No file uploaded."


if __name__ == '__main__':
    app.run(debug=True)

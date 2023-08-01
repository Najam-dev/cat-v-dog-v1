import tensorflow as tf 
import cv2
from tensorflow import keras
# import matplotlib.pyplot as plt


model = tf.keras.models.load_model("model.h5")

test_img = cv2.imread('doggg.jpg')

# plt.imshow(test_img)
test_img = cv2.resize(test_img,(256,256))
test_input = test_img.reshape((1,256,256,3))
test_input = test_input / 255.0  # Normalize pixel values to [0, 1]

# Make the prediction
prediction = model.predict(test_input)
per = float(prediction * 100)
print (per)

# Convert the prediction to a percentage
per_dog = float(prediction[0][0] * 100)
per_cat = float((1 - prediction[0][0]) * 100)

# Print the percentage confidence for both "dog" and "cat" predictions
print("The model predicts the image is {:.2f}% dog and {:.2f}% cat.".format(per_dog, per_cat))

# # Interpret the prediction result
# if prediction[0][0] >= 0.5:
#     print("The model predicts the image is a {:.2f}% dog.".format(per))
# else:
#     print("The model predicts the image is a {:.2f}% cat.".format(per))


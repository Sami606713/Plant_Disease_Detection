from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np

def load_save_model():
    model=load_model('model/plant_disease.keras')

    return model

def process_image(img):
    # Convert the image to RGB and resize it
    img = Image.open(img)

    img = img.convert('RGB')
    img = img.resize((224, 224))
    
    # Convert the image to an array
    image_array = image.img_to_array(img)

    # Expand the dimensions of the image array to add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Rescale the image array values to be between 0 and 1
    image_array = image_array / 255.0

    return image_array

def predict(image):
    # load the processor to process the image
    image_array=process_image(img=image)

    # load the model
    model = load_save_model()

    # Prediction
    prediction=np.argmax(model.predict(image_array),axis=1)

    return prediction


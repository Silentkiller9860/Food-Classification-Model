import streamlit as st 
import os
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import keras_preprocessing as kp
import requests
from bs4 import BeautifulSoup as bs


st.title("Food Classification Demo")
model=load_model('Model Path')

def predict_calories(prediction):
    url = 'https://www.google.com/search?&q=calories+in  '+prediction
    req = requests.get(url).text
    scrap = bs(req, 'html.parser')
    calories = scrap.find('div', class_="BNeawe iBp4i AP7Wnd").text
    return calories



def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False



category={0: 'Burger', 1: 'Butter Naan', 2:'Chai',
    3:'Chapati', 4:'Chole Bhature', 5:'Dal Makhani',
    6: 'Dhokla', 7:'Fried Rice', 8:'Idli', 9:'Jalebi',
    10: 'Kaathi Rolls', 11: 'Kadai Paneer', 12: 'Kulfi',
    13: 'Masala Dosa', 14:'Momos', 15:'Paani Puri',
    16:'Pakode', 17:'Pav Bhaji', 18:'Pizza', 19:'Samosa'
}


def predict_image(filename , model):
  img_ = kp.image.load_img(filename , target_size = (299,299))
  img_array = kp.image.img_to_array(img_)
  img_processed = np.expand_dims(img_array , axis = 0)
  img_processed /= 255.
    
  prediction = model.predict(img_processed)
  index = np.argmax(prediction)
  display_img = Image.open(uploaded_image)
  pred = category[index]
  st.header("Prediction - {}".format(pred))

  st.info(f"Calories in {pred} are {predict_calories(pred)} in 100 grams")
  st.image(display_img)

    

Food = st.selectbox('Select Food Item',category.values())
st.text(f"Please upload Image of {Food} to give to Machine Learning Algorithm .")

uploaded_image = st.file_uploader('Choose an image')

save_uploaded_image(uploaded_image)
predict_image(os.path.join('uploads',uploaded_image.name),model)

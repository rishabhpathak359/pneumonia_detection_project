import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

model = load_model()

st.set_page_config(page_title='Pneumonia Detection', layout='centered')
st.title('Pneumonia Detection from Chest X-Ray')

st.sidebar.header('Upload')
uploaded_file = st.sidebar.file_uploader('Choose a chest X-ray image', type=['png','jpg','jpeg'])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray', use_column_width=True)
    st.write('')
    st.write('Classifying...')
    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    score = float(prediction)
    if score > 0.5:
        st.error(f'Pneumonia detected with confidence {score:.2f}')
    else:
        st.success(f'Normal lung detected with confidence {1-score:.2f}')

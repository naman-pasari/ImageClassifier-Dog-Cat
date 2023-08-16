import gradio as gr
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model

labels = ['Cat','Dog']

def predict(image):
    
        resize = tf.image.resize(image, (256,256))
        new_model = load_model(os.path.join(os.getcwd(),'notebook','models','imageclassifier.h5'))
        prediction=new_model.predict(np.expand_dims(resize/255, 0))
        if prediction > 0.5: 
            return labels[1]
        else:
            return labels[0]
        
image = gr.inputs.Image(shape=(256,256))
label = gr.outputs.Label(num_top_classes=2)

gr.Interface(fn=predict,inputs=image, outputs=label, capture_session=True).launch()
import tensorflow as tf
import numpy as np
import gradio as gr
import json
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("model/model.h5")

with open("model/labels.json") as f:
    labels = json.load(f)
labels = {int(k): v for k, v in labels.items()}

def predict(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)

    return f"{labels[idx]} ({preds[idx]*100:.2f}%)"

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Apple Disease Classifier"
).launch()

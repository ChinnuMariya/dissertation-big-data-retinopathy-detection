
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = FastAPI()

model_d = tf.keras.models.load_model(r"C:\Users\ANAKHA\Downloads\Aptos1\saved_models\model_d.keras")
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def preprocess_image(img: Image.Image, target_size=(180, 180)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = preprocess_image(img)
    prediction = model_d.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    import base64
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        img.save(tmp.name)
        with open(tmp.name, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": predicted_class,
        "image_data": image_data
    })

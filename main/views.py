from django.shortcuts import render
from .form import ImageForm
from .models import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np


def index(request):
    if request.method == "POST":
        form = ImageForm(data=request.POST, files=request.FILES)

        if form.is_valid():
            form.save()
            obj = form.instance
            predict()
            return render(request, "main/index.html", {"obj": obj})
    else:
        form = ImageForm()
    img = Image.objects.all()
    return render(request, "main/index.html", {"img": img, "form": form})


def predict():
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    model = keras.models.load_model('CNN')
    image_size = (32, 32)
    url = "https://karibay-image-classification.herokuapp.com\media\{url}".format(url=Image.objects.latest('pk').image)

    img = keras.preprocessing.image.load_img(url, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = predictions[0]
    class_index = np.argmax(score)

    last_image = Image.objects.latest('pk')
    last_image.caption = classes[class_index]
    last_image.save()
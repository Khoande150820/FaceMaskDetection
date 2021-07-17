import cv2
import numpy as np
from tensorflow.keras import models
from keras.models import load_model
FRONTAL_FACES = 'haarcascades/haarcascade_frontalface_default.xml'
face_model = cv2.CascadeClassifier(FRONTAL_FACES)
model = load_model("VGG19-Face Mask Detection.h5")
path = "image/__results___23_0.png"


mask_label = {0:'MASK',1:'NO MASK'}
dist_label = {0:(0,255,0),1:(255,0,0)}


def predict_image(image_dir):
    while True:
        img = cv2.imread(image_dir)
        img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

        # returns a list of (x,y,w,h) tuples
        faces = face_model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

        out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


        for i in range(len(faces)):
            (x, y, w, h) = faces[i]
            crop = out_img[y:y + h, x:x + w]
            crop = cv2.resize(crop, (128, 128))
            crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
            mask_result = model.predict(crop).argmax()
            cv2.rectangle(out_img, (x, y), (x + w, y + h), dist_label[mask_result], 1)
            cv2.putText(
                img, mask_label[mask_result],
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break


predict_image(path)

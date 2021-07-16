import cv2
import numpy as np
from tensorflow.keras import models

FRONTAL_FACES = 'haarcascades/haarcascade_frontalface_default.xml'

model = models.load_model("model")
path = "image/giroudDiscord.JPG"

def result(img):
    img = cv2.resize(img, (128, 128))
    img = np.reshape(img, (1, 128, 128, 3)) / 255.0
    if ((model.predict(img) > 0.5).astype("int32")) == 0:
        return "Have Mask"
    else:
        return "No mask"


def detect_mask(img_path):
    face_cascades = cv2.CascadeClassifier(FRONTAL_FACES)
    cap = cv2.VideoCapture(0)
    while True:
        # ret, img = cap.read()
        img = cv2.imread(img_path)
        if img is None:
            print("Can't find image")
            return None
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascades.detectMultiScale(gray, 1.1, 3)
            if faces is None:
                print("No face detected!")
                return 0
            else:
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face_detected = img[y:y + h, x:x + w]
                    print(result(face_detected))
                    cv2.putText(img, result(face_detected), (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("img", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break



detect_mask(path)

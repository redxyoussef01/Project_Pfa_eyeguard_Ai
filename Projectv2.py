import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from io import BytesIO
import time

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt

from os import listdir

import firebase_admin
from firebase_admin import credentials, storage, db


cred = credentials.Certificate(
    r'D:\\Python project Face\\face-recognation-6ecd1-firebase-adminsdk-pnzvl-82505c31e0.json')
firebase_admin.initialize_app(
    cred, {'storageBucket': 'face-recognation-6ecd1.appspot.com', 'databaseURL': 'https://face-recognation-6ecd1-default-rtdb.firebaseio.com/'})

bucket = storage.bucket()
ref = db.reference('personne_personne')

ref2 = db.reference('authorisation_autorisation')
color = (67, 67, 67)

face_cascade = cv2.CascadeClassifier(
    'C:/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
)


def preprocess_image(image_data):
    img = Image.open(BytesIO(image_data))
    # Resize image to match the model's input shape
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def loadVggFaceModel():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    from keras.models import model_from_json
    model.load_weights(
        r'D:\Python project Face\vgg_face_weights.h5')

    vgg_face_descriptor = Model(
        inputs=model.layers[0].input, outputs=model.layers[-2].output)

    return vgg_face_descriptor


model = loadVggFaceModel()

employees = {}
employeesID = {}
employees_autorisation = {}
# Get the data from the Realtime Database
employees_data = ref.get()
employees_autori = ref2.get()

if employees_autori is not None:
    for employee_id, employee_auto in employees_autori.items():
        if all(key in employee_auto for key in ['authorized', 'id']):
            empid = employee_auto['id']
            employee_state = employee_auto['authorized']
            employees_autorisation[empid] = employee_state
        else:
            print(f"Missing data for employee with ID: {employee_id}")

# Check if the data is not None
if employees_data is not None:
    for employee_id, employee_info in employees_data.items():
        # Check if the employee_info contains 'nom', 'prenom', and 'id' fields
        if all(key in employee_info for key in ['nom', 'prenom', 'id']):
            nom = employee_info['nom']
            prenom = employee_info['prenom']
            employee_id = employee_info['id']
            full_name = nom + " " + prenom
            # Construct the image name
            image_name = f"{employee_id}_{nom}_{prenom}"
            try:
                # Download the image from the Google Cloud Storage bucket
                img_data = bucket.blob(
                    f'database/{image_name}.jpg').download_as_string()
                img = preprocess_image(img_data)  # Preprocess the image
                employees[full_name] = model.predict(img)[0, :]
                employeesID[full_name] = employee_id

            except:
                try:
                    # If .jpg download fails, try .JPG extension
                    img_data = bucket.blob(
                        f'database/{image_name}.JPG').download_as_string()
                    img = preprocess_image(img_data)  # Preprocess the image
                    employees[full_name] = model.predict(img)[0, :]
                    employeesID[employee_id] = full_name
                except Exception as e:
                    print(f"Error downloading image for {image_name}: {e}")

else:
    print("No data found in the Realtime Database")


def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


password = "zhxh oynh hsnp rnut"
# must match the email used to generate the password
from_email = "yourais04@gmail.com"
to_email = "yourais00@gmail.com"

server = smtplib.SMTP('smtp.gmail.com: 587')
server.starttls()
server.login(from_email, password)


def send_email(to_email, from_email, object_detected=1, frame_image=None):
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = "Security Alert"

    # Add in the message body
    message_body = f'ALERT - {object_detected} Person has been detected Smoking!!'
    message.attach(MIMEText(message_body, 'plain'))

    # Attach the frame image to the email
    if frame_image is not None:
        _, image_buffer = cv2.imencode('.jpg', frame_image)
        attachment = MIMEImage(image_buffer.tobytes())
        attachment.add_header('Content-Disposition',
                              'attachment', filename="frame.jpg")
        message.attach(attachment)

    # Send the email
    server.sendmail(from_email, to_email, message.as_string())


class ObjectDetection:
    def __init__(self, capture_index):
        # default parameters
        self.capture_index = capture_index
        self.email_sent = False

        # model information
        self.model = YOLO("best.pt")

        # visual information
        self.annotator = None
        self.start_time = 0
        self.end_time = 0

        # device information
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Delay parameters
        self.email_delay = 60  # Delay in seconds
        self.last_email_time = 0  # Time when the last email was sent

    def predict(self, im0):
        results = self.model(im0)
        return results

    def display_fps(self, im0):
        self.end_time = time.time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(im0, (20 - gap, 70 - text_size[1] - gap),
                      (20 + text_size[0] + gap, 70 + gap), (255, 255, 255), -1)
        cv2.putText(im0, text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0, employees):
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            if cls == 0:  # Assuming class ID 0 corresponds to faces
                # Crop detected face
                detected_face = im0[int(box[1]):int(
                    box[3]), int(box[0]):int(box[2])]
                detected_face = cv2.resize(
                    detected_face, (224, 224))  # Resize to 224x224

                # Preprocess face image for recognition
                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 127.5
                img_pixels -= 1

                # Predict face representation
                captured_representation = model.predict(img_pixels)[0, :]

                # Perform face recognition
                found = False
                for employee_name, representation in employees.items():
                    similarity = findCosineSimilarity(
                        representation, captured_representation)
                    if similarity < 0.35:  # Threshold for similarity
                        label = employee_name
                        found = True

                        print(employeesID[employee_name])
                        print(
                            employees_autorisation[employeesID[employee_name]])
                        break
                if not found:
                    label = 'Unknown'

                # Draw label on image
                self.annotator.box_label(
                    box, label=label, color=colors(int(cls), True))
            else:
                # Use YOLO class names for other objects
                label = names[int(cls)]
                self.annotator.box_label(
                    box, label=label, color=colors(int(cls), True))
        return im0, class_ids

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count = 0
        while True:
            self.start_time = time.time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0, employees)

            if 1 in class_ids:  # Only send email if "smoke" class is detected
                if not self.email_sent:
                    # Check if enough time has passed since the last email
                    current_time = time.time()
                    if current_time - self.last_email_time >= self.email_delay:
                        # Send email with the frame image
                        send_email(to_email, from_email,
                                   class_ids.count(1), im0)
                        self.email_sent = True
                        self.last_email_time = current_time  # Update last email time
                else:
                    self.email_sent = False

            cv2.imshow('Eye Guard', im0)
            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:  # press q to quit
                break

        cap.release()
        cv2.destroyAllWindows()


detector = ObjectDetection(capture_index=0)
detector()

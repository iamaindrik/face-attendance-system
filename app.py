import cv2
import os
from flask import Flask, request, render_template, redirect, url_for
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)

nimgs = 10
imgBackground = cv2.imread("background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

attendance_running = False

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time\n')


def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'{username},{userid},{current_time}\n')


@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/start', methods=['GET'])
def start():
    global attendance_running
    attendance_running = True

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return redirect(url_for('home', mess='Model not found. Add a new face.'))

    cap = cv2.VideoCapture(0)
    while attendance_running:
        ret, frame = cap.read()
        if ret:
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))
                # KNOWN
                if identified_person.size > 0:
                    identified_person = identified_person[0]

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for known faces
                    cv2.putText(frame, identified_person, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (255, 0, 0), 2)
                    add_attendance(identified_person)
                else:
                    # UNKNOWN
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for unknown faces
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (255, 0, 0), 2)

            imgBackground[162:642, 55:695] = frame
            cv2.imshow('Attendance', imgBackground)

            if cv2.waitKey(1) == 27:  # Stop on 'Esc' key
                break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('home'))


@app.route('/stop', methods=['GET'])
def stop():
    global attendance_running
    attendance_running = False
    return redirect(url_for('home'))


@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'

    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    cap = cv2.VideoCapture(0)
    i, j = 0, 0

    while j < nimgs * 5:
        ret, frame = cap.read()
        faces = extract_faces(frame)

        for (x, y, w, h) in faces:
            if j % 5 == 0:
                img_name = f'{newusername}_{i}.jpg'
                cv2.imwrite(f'{userimagefolder}/{img_name}', frame[y:y + h, x:x + w])
                i += 1
            j += 1

        cv2.imshow('Adding New User', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    train_model()

    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)

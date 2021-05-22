from flask import Flask, render_template, request, redirect, session
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import mysql.connector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
app = Flask(__name__)
app.secret_key = os.urandom(24)

conn = mysql.connector.connect(
    host="remotemysql.com", user="MnCD7IeHYs", password="nGQ2tXRvzo", database="MnCD7IeHYs")
cursor = conn.cursor()


@app.route('/')
def login():
    return render_template('login.html')


@app.route('/register')
def about():
    return render_template('register.html')


@app.route('/home')
def home():
    if 'user_id' in session:
        return render_template('home.html')
    else:
        return redirect('/')


@app.route('/login_validation', methods=['POST'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')
    cursor.execute(
        """SELECT *  FROM `users` WHERE `email` LIKE '{}' AND `password` LIKE '{}'""".format(email, password))
    users = cursor.fetchall()
    if len(users) > 0:
        session['user_id'] = users[0][0]
        return redirect('/home')
    else:
        return redirect('login.html')


@app.route('/camera')
def index2():

    face_classifier = cv2.CascadeClassifier(
        r'D:\NITEESH\studies\CBIT\4\4.2\ClassGenie\haarcascade_frontalface_default.xml')
    classifier = load_model(
        r'D:\NITEESH\studies\CBIT\4\4.2\ClassGenie\model.h5')

    emotion_labels = ['concentrated', 'Disgust', 'sleepy',
                      'intrested', 'calm', 'not_intrested', 'sleepy']

    cap = cv2.VideoCapture(0)

    i = 0

    while os.path.exists(f"file_{i}.txt"):
        i += 1

    while True:

        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48),
                                  interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]

                label = emotion_labels[prediction.argmax()]
                file = open(f"file_{i}.txt", "a")
                file.write(label + '\n')

                label_position = (x, y-10)
                cv2.putText(frame, label, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Attention Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    file.close()
    cap.release()
    cv2.destroyAllWindows()
    return render_template('home.html')


@app.route('/cameraoff')
def test():
    i = 0
    j = 0
    while os.path.exists(f"file_{i}.txt"):
        i += 1
    while os.path.exists(f"outfile_{j}.txt"):
        j += 1

    sys.stdout = open(f"outfile_{j}.txt", "a")

    file = open(f"file_{i-1}.txt", "r")

    d = dict()

    for line in file:
        line = line.strip()

        line = line.lower()
        words = line.split(" ")
        for word in words:
            if word in d:
                d[word] = d[word] + 1
            else:
                d[word] = 1

    for key in list(d.keys()):
        print(key, ":", d[key])

    file.close()
    sys.stdout.close()

    return render_template('home.html')


@app.route('/displaydata')
def Graph():
    z = 0

    expression = []
    numberoftimes = []

    while os.path.exists(f"outfile_{z}.txt"):
        z += 1

    file = open(f"outfile_{z-1}.txt", 'r')

    for row in file:
        row = row.split(' : ')
        expression.append(row[0])
        numberoftimes.append(int(row[1]))

    plt.bar(expression, numberoftimes, color='g', label='File Data')

    plt.xlabel('Expression', fontsize=12)
    plt.ylabel('TIMES', fontsize=12)

    plt.title('Attention', fontsize=20)
    plt.savefig(
        r"D:\NITEESH\studies\CBIT\4\4.2\ClassGenie\web\static\barchart.png")
    return render_template('/data.html')


@app.route('/add_user', methods=['POST'])
def add_user():
    user_id = request.form.get('user_id')
    name = request.form.get('uname')
    email = request.form.get('uemail')
    password = request.form.get('upassword')

    cursor.execute("""INSERT INTO `users` (`user_id`,`name`,`email`,`password`)VALUES
    ('{}','{}','{}','{}')""".format(user_id, name, email, password))
    conn.commit()
    cursor.execute(
        """SELECT * FROM `users` WHERE `email` LIKE '{}'""".format(email))
    myuser = cursor.fetchall()
    session['user_id'] = myuser[0][0]
    return redirect('/home')


@app.route('/logout')
def logout():
    session.pop('user_id')
    redirect('/')


if __name__ == "__main__":
    app.run(debug=True)

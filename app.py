import tensorflow as tf
from keras.models import load_model

from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array 
from keras_preprocessing import image
import cv2
import numpy as np
import pandas as pd
from music import music_rec
from flask import Flask, render_template, Response, jsonify
app = Flask(__name__)
face_clasifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
clasifier = load_model('./model/model2.h5')
show_text=[0]
class_labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise' }
music_dist={0:"songs/angry.csv",1:"songs/disgusted.csv ",2:"songs/fearful.csv",3:"songs/happy.csv",4:"songs/neutral.csv",5:"songs/sad.csv",6:"songs/surprised.csv"}
headings = ("Name","Album","Artist")
df1 = music_rec()
df1 = df1.head(15)

@app.route('/')
def index():
    print(df1.to_json(orient='records'))
    return render_template('index.html', headings=headings, data=df1)

def process():
    global df1
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_clasifier.detectMultiScale(gray, 1.3, 5)
        df1 = pd.read_csv(music_dist[show_text[0]])
        allfaces=[]
        rects=[]
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            rol_gray = gray[y:y+h, x:x+w]
            rol_gray = cv2.resize(rol_gray, (48,48), interpolation=cv2.INTER_AREA)
            allfaces.append(rol_gray)
            rects.append((x, w, y, h))
        i = 0
        for face in allfaces :
            rol = face.astype("float") / 255.0
            rol = img_to_array(rol)
            rol = np.expand_dims(rol, axis=0)
            preds = clasifier.predict(rol)[0]
            label = class_labels[preds.argmax()]
            show_text[0] = preds.argmax()
            music = pd.read_csv("music.csv")
            label_music = music.loc[(music['mood'] == label), "name"]
            dict_music = label_music.to_list()
            
            label_posisition = (rects[i][0] + int((rects[i][1]/2)), abs(rects[i][2] - 10))
            label_posisition2 = ((rects[0][0]), rects[i][2] + rects[i][3] + 20)
            i = + 1
            cv2.putText(frame, label, label_posisition , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(dict_music), label_posisition2 , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            df1 = music_rec()
        #cv2.imshow("Emotion Detection", img)
        ret, buffer = cv2.imencode('.jpg', frame) #compress and store image to memory buffer
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') #concat frame one by one and return frame

        if cv2.waitKey(1) == 13:
            break
    cap.release()
    cv2.destroyAllWindows()



@app.route('/video_feed')
def video_feed():
    #Video streaming route
    return Response(process(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/t')
def gen_table():
    return df1.to_json(orient='records')

if __name__ == "__main__":
    app.run(debug=True)
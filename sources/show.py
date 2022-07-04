import tkinter as tk
from tkinter import *
import imageio
import cv2
from PIL import Image, ImageTk
import os
from tkinter.filedialog import askopenfile 
from tkinter import filedialog
from tkinter import Canvas
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import threading
import time
import os
import pathlib
from functools import partial
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.load_weights('model/emotion_model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}
gender_labels = ['Male', 'Female']

emoji_dist={0:"files/emojis/angry.png",1:"files/emojis/disgusted.png",2:"files/emojis/fearful.png",3:"files/emojis/happy.png",4:"files/emojis/neutral.png",5:"files/emojis/sad.png",6:"files/emojis/surprised.png"}

car_dist={0:"files/car/poza1.png", 1:"files/car/poza2.png", 2:"files/car/poza3.png"}
global last_frame1   
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
cap = cv2.VideoCapture('files/videos/angry_video.mp4')
global frame_number
show_text=[0]

 

def test():
    global lmain, lmain2, lmain3,lmain4, root,btn, canvas
    root = tk.Tk()
    #btn = Button(root, text = 'Click me !', command = lambda:video(), font=('arial',25,'bold'))
    all_btn = Button(text="All",command=partial(change_value, 'files/videos/video_slow.mp4'), width=35, height=1, font=('arial',8,'bold'), bd=4, bg='black', fg= "white")
    angry_btn = Button(text="Angry",command=partial(change_value, 'files/videos/happy_video.mp4'), width=35, height=1, bd=4, bg='black', fg= "white")
    disgusted_btn = Button(text="Disgusted",command=partial(change_value, 0), width=35, height=1, bd=4, bg='black', fg= "white")
    fearful_btn = Button(text="Fearful",command=change_value, width=35, height=1, bd=4, bg='black', fg= "white")
    happy_btn = Button(text="Happy",command=happy_pressed, width=35, height=1, bd=4, bg='black', fg= "white")
    neutral_btn = Button(text="Neutral",command=change_value, width=35, height=1, bd=4, bg='black', fg= "white")
    sad_btn = Button(text="Sad",command=change_value, width=35, height=1, bd=4, bg='black', fg= "white")
    surprised_btn = Button(text="Surprised",command=change_value, width=35, height=1, bd=4, bg='black', fg= "white")
    video_btn = Button(text="Video",command=change_value, width=35, height=1, bd=4, bg='black', fg= "white")
    #place the buttons
    all_btn.pack()
    all_btn.place(x=920, y=530)
    angry_btn.pack() 
    angry_btn.place(x=920, y=570)
    disgusted_btn.pack()
    disgusted_btn.place(x=920, y=610)
    fearful_btn.pack()
    fearful_btn.place(x=920, y=650)
    happy_btn.pack()
    happy_btn.place(x=920, y=690)
    neutral_btn.pack()
    neutral_btn.place(x=920, y=730)
    sad_btn.pack()
    sad_btn.place(x=920, y=770)
    surprised_btn.pack()
    surprised_btn.place(x=920, y=810)
    video_btn.pack()
    video_btn.place(x=920, y=850)
    lmain = tk.Label(master=root, padx=50, bd=2) #video
    lmain2 = tk.Label(master=root, borderwidth=3, relief="flat") #avatar
    lmain3 = tk.Label(master=root,fg="#CDCDCD", bg='black', highlightcolor="black",font=('arial',12,'bold')) #scris
    lmain4 = tk.Label(master=root,bd=2) #masina
    Label(root, text="Menu", bg="black", fg="white", font=('arial',25,'bold')).place(x=1000, y=20)
    Label(root, text="Current state:", bg="black", fg="white", font=('arial',25,'bold')).place(x=710, y=400)
    lmain.pack(side=LEFT)
    lmain.place(x=70,y=50)
    lmain4.pack(side=LEFT)
    lmain4.place(x=70,y=450)
    lmain3.pack()
    lmain3.place(x=950,y=395)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=920,y=100)
    canvas=Canvas(master=root, width=1, height=900)
    canvas.pack()

    root.title("Mood Tracker")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'

    root.mainloop()


def run():
    while True:
        print('thread running')
        global stop_threads
        if stop_threads:
            break

value = 1
def happy_pressed():
        print("button pressed")
        time.sleep(1)
        stop_threads = True
        
def angry_pressed():
        time.sleep(1)
        stop_threads = True
        threading.Thread(target=show_subject, daemon=True, args= ("files/videos/angry.mp4",)).start()
        threading.Thread(target=show_avatar, daemon=True).start()
        value = 1


def change_value(value):
        global cap
        if threading.active_count() == 1:
            cap = cv2.VideoCapture(value)
            threading.Thread(target=show_subject, daemon=True, args=()).start()
            threading.Thread(target=show_avatar, daemon=True).start()
        else:
            cap = cv2.VideoCapture(value)

        #threading.Thread(target=show_avatar, daemon=True).start()

def show_subject():  
    global cap
    #cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        
        if not ret:
            break
           
        frame = cv2.resize(frame,(500,400))
        facecasc = cv2.CascadeClassifier('files/libs/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            show_text[0]=maxindex
        
        #cv2.imshow('Video', cv2.resize(frame,(1000,960),interpolation = cv2.INTER_CUBIC))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)
        lmain.configure(image=frame)
        lmain.image = frame
        #frame3=cv2.imread(car_dist[0])
        if maxindex==5:
            frame3=cv2.imread(car_dist[0])
        else:
            frame3=cv2.imread(car_dist[1])

        frame3=cv2.cvtColor(frame3,cv2.COLOR_BGR2RGB)
        frame3 = cv2.resize(frame3,(500,400))
        frame3=Image.fromarray(frame3)
        frame3=ImageTk.PhotoImage(image=frame3)
        #lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',30,'bold'))
        lmain4.configure(image=frame3)
        lmain4.frame3=frame3


    cap.release()
    cv2.destroyAllWindows()



def show_avatar():
    print('============================================================SHOW TEXT DE 0 ' + str(show_text[0]))
    print('============================================================EMOJI_DIST DE SHOW TEXT ' + emoji_dist[show_text[0]])


    frame2=cv2.imread(emoji_dist[show_text[0]])
    frame2 = cv2.resize(frame2,(250,250))
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(frame2)
    imgtk2=ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2=imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',30,'bold'))
    lmain2.configure(image=imgtk2)                                                
    lmain2.after(5, show_avatar)

    


if __name__ == '__main__':
    frame_number = 0
    test()
    #change_value()
    root.mainloop()

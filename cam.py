import cv2
import  numpy as np
from matplotlib import pyplot as plt

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#print facec
font = cv2.FONT_HERSHEY_SIMPLEX

f=np.load('pic_data/my_face_data.npy')
f=f.reshape(f.shape[0],f.shape[1]*f.shape[2])

labels=np.load('pic_data/my_face_labels.npy')

def dist(x1,x2):
    return np.sqrt(((x1-x2)**2).sum());

def knn(X_train,x,y_train,k=5):
    vals=[]
    for ix in range(X_train.shape[0]):
        v=[dist(x,X_train[ix,:]), y_train[ix]]
        vals.append(v)
    updated_vals=sorted(vals,key=lambda x:x[0])
    pred_arr=np.asarray(updated_vals[:k])
    pred_arr=np.unique(pred_arr,return_counts=True)
    pred=pred_arr[1].argmax()
    return pred_arr[0][pred]

def get_name(im):
    return knn(f, im, labels, k=5)

def recognize_face(im):
    im = cv2.resize(im, (100, 100))
    im = im.flatten()
    return get_name(im)

while True:
    _, fr = rgb.read()
    
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    #print gray.shape
    faces = facec.detectMultiScale(gray, 1.3, 5)
    print len(faces)
    
    for (x,y,w,h) in faces:
	#print x,y,w,h
        fc = fr[x:x+w, y:y+h, :]
	print fc.shape
        out = recognize_face(cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY))
	#out = 'Hey'
        cv2.putText(fr, out, (x, y), font, 1, (255, 255, 0), 2)
    	cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('rgb', fr)
    cv2.imshow('gray', gray)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

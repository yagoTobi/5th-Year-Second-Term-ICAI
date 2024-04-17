# MAIN
import cv2
import imutils
from keras.models import load_model
from PIL import Image, ImageDraw
import os
import numpy as np
camera = cv2.VideoCapture(0)
#model = load_model("Twentyfive_class.model")
IMG_SIZE = 100
top, right, bottom, left = 10, 340, 235, 600  # Adjusted values
_,first_frame = camera.read()

SAVE_PATH = "C:/Users/Admin/Downloads/Compressed"
#camera.release()
first_gray = cv2.flip(first_frame, 1)
roi = first_gray[top:bottom, right:left]
img_count = 1
dictonary = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 'M':11, 'N':12, 'O':13, 'P':14, 
            'Q':15, 'R':16, 'S':17, 'T':18, 'U':19, 'V':20, 'W':21, 'X':22, 'Y':23}
num_class = 24
out = ""
inv_dictonary = dict(map(reversed, dictonary.items()))
string = ""
while True:
    
    _, frame = camera.read()
    
    gray_frame = cv2.flip(frame, 1)
    
    #gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
    #gray_frame = cv2.GaussianBlur(gray_frame, (35,35), 0)
    
    roi2 = gray_frame[top:bottom, right:left]
    cv2.rectangle(gray_frame, (left, top), (right, bottom), (0,255,0), 2)
    
    diff = cv2.absdiff(roi, roi2)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.GaussianBlur(diff, (11,11),0)
    _,diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    
    cv2.imshow("Diff", diff)
    
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == ord('q'):
        break
    if key == ord('r'):
        roi = gray_frame[top:bottom, right:left]
        string = ""
    
    # PREDICTION
    
    #img = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(diff, (IMG_SIZE, IMG_SIZE))
    
    img_arr = np.array(img)

    t2 = np.expand_dims(img_arr, axis=-1)
    t2 = np.expand_dims(t2, axis=0)

    #p = model.predict(t2)
    #p2 = np.argmax(p)
    #out = inv_dictonary[p2]
    
    #PREDICTION ENDS
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(gray_frame,out, (22,34), font, 1, (200,255,255), 2, cv2.LINE_AA)
    cv2.putText(gray_frame,string, (22,64), font, 1, (200,255,255), 2, cv2.LINE_AA)
    #cv2.putText(gray_frame,"r: Reset", (22,400), font, 1, (200,255,255), 2, cv2.LINE_AA)
    #cv2.putText(gray_frame,"s: Append word", (22,430), font, 1, (200,255,255), 2, cv2.LINE_AA)
    cv2.putText(gray_frame,"r: Reset;  s: Append;  q: Quit", (22,470), font, 1, (200,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow("Frame", gray_frame)
    if key == ord('s'):
        string = string+""+out
    
    #cv2.imshow("Frame", gray_frame)
    
    if key == ord('c'):
        img_name = "data{}.png".format(img_count)
        img_count+=1
        cv2.imwrite(os.path.join(SAVE_PATH, img_name), diff)
        
        img_name = "data{}.png".format(img_count)
        img_count+=1
        cv2.imwrite(os.path.join(SAVE_PATH, img_name), gray_frame)
        
        img_name = "data{}.png".format(img_count)
        img_count+=1
        cv2.imwrite(os.path.join(SAVE_PATH, img_name), roi2)
        #cv2.imwrite(img_name, roi2)
        
camera.release()
cv2.destroyAllWindows()
import cv2 as cv
import os


os.chdir("R:/Face/model/validation")
#change validation to train for capturing training images
n=input('Enter your name to add a new face')
cam=cv.VideoCapture(0)
img_no=0

while True:
    result,image=cam.read()
    img_no+=1
    face=cv.resize(image, (200,200))
    #face=cv.cvtColor(face,cv.COLOR_BGR2GRAY)
    path='saji/'+n+str(img_no)+'.jpg'
    cv.imwrite(path,face)
    cv.imshow('image',face)
    print(img_no)
    if cv.waitKey(1)=='q' or int(img_no)==1400:
      #change number of photos accordingly
        break
cv.destroyWindow('image')
del cam

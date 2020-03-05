# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:32:39 2020

@author: 33672
"""

import numpy as np
import cv2
import glob



def resize(img):
    try:
        #Variables
        width = 600
        height = 400
        dim = (width, height)
        #Resize
    
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    except ValueError:
        pass
    return img

def cascade_haar(frame):
# Variables --------------------------------
    

#-------------------------------------------
    
    cascade = cv2.CascadeClassifier('cascade xml\cascadePassage.xml')
    crosswalk = cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x,y,w,h) in crosswalk: 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
    return frame


def feu_vert(img):
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        ## (2) Find the target green color region in HSV
        hsv_lower = np.array([50, 104, 141])
        hsv_upper = np.array([86, 245, 245])
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        
        ## (3) morph-op to remove horizone lines
        kernel = np.ones((5,1), np.uint8)
        mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        
        #cv2.waitKey(0)
        ## (4) crop the region
    
        ys, xs = np.nonzero(mask2)
    
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()
        
        
        
        #croped = img[ymin:ymax, xmin:xmax]
        
        pts = np.int32([[xmin, ymin],[xmin,ymax],[xmax,ymax],[xmax,ymin]])
        cv2.drawContours(img, [pts], -1, (0,255,0), 1, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Feu Vert',(xmin, ymin-10), font, 1,(255,255,255),2,cv2.LINE_AA)
        
        
        return img, 1

    except ValueError:
        pass
    
    return img, 0
        



def feu_rouge(img):
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        ## (2) Find the target red color region in HSV
        hsv_lower = np.array([123,144,197])
        hsv_upper = np.array([180,255,255])
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        
        ## (3) morph-op to remove horizone lines
        kernel = np.ones((5,1), np.uint8)
        mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        
        #cv2.waitKey(0)
        ## (4) crop the region
    
        ys, xs = np.nonzero(mask2)
    
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()
        
        
        
        #croped = img[ymin:ymax, xmin:xmax]
        
        pts = np.int32([[xmin, ymin],[xmin,ymax],[xmax,ymax],[xmax,ymin]])
        cv2.drawContours(img, [pts], -1, (0,0,255), 1, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Feu rouge',(xmin, ymin-10), font, 1,(255,255,255),2,cv2.LINE_AA)
        
        
        return img, 1

    except ValueError:
        pass
    
    return img, 0





def passage_pieton(frame, wait, secure, seconde, count):
    frame_original = frame
    try:
        
        count = 0
        wait = wait - 1
        seconde = (seconde + 1)%60
        activator = 0
    #---------------------------------
        

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame = cv2.GaussianBlur(frame,(3,3),cv2.BORDER_DEFAULT)
        
        #_,frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
        canny = cv2.Canny(frame,100,255)
        
        frame = canny
    
        contours,_=cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours : 
            area = cv2.contourArea(cnt) 
       
            if (area > 400 and area < 5000):  
                approx = cv2.approxPolyDP(cnt,  0.01 * cv2.arcLength(cnt, True), True) 
            
                if(len(approx) == 4): 
                    cv2.drawContours(canny, [approx], 0, (255, 0, 0), 2)
                    count = count + 1
        if(count >= 3):
            wait = 60
            secure = secure + 1
            seconde = 1
        if(seconde == 59):
            secure = 0
        if((count >= 3 or wait >= 0) and secure >= 6 ):
            activator = 1
            cv2.putText(frame_original,'Passage pieton', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
             
    except ValueError:
        pass  
    return frame_original, wait, secure, seconde, count, activator

#-----------------------------------------
#                 MAIN
#-----------------------------------------
    
# TEXTE Constantes ----------------------- 
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,380)
bottomRightCornerOfText = (400,380)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
#-----------------------------------------

# Variables ------------------------------
wait = 0 
secure = 0
seconde = 0
count = 0   
delay_vert = 0
delay_rouge = 0
#-----------------------------------------
   


a = input('Appuyez sur "v" pour analyser des video, sinon appuyez sur "entrée"\n\nVotre choix ("entrée" pour valider):')

if (a=='v'):
    for video_path in glob.glob('video\*.mp4'):
        cap = cv2.VideoCapture(video_path)
        #cap = cv2.VideoCapture(0)
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            frame = resize(frame)
            if ret == True:
# Appeler fonctions ici ---------------------------------------------
# Argument = frame
                #cascade_haar(frame) # Solution non viable car peu performante
                frame, wait, secure, seconde, count, activator_pp = passage_pieton(frame, wait, secure, seconde, count)
                frame, activator_fv = feu_vert(frame)
                frame, activator_fr = feu_rouge(frame)
                # SOME DELAY 
                if((activator_pp == 1 and activator_fv == 1) and delay_vert <= 0):
                    delay_vert = 30
                if((activator_pp == 1 and activator_fr == 1) and delay_rouge <= 0):
                    delay_rouge = 30

                    
                # CONDITIONS POUR TRAVERSER
                if((activator_pp == 1 and activator_fv == 1) or delay_vert >= 0):
                    delay_vert = delay_vert - 1
                    cv2.putText(frame,'TRAVERSEZ', bottomRightCornerOfText, font, fontScale,(0, 255, 0),lineType)    
                elif((activator_pp == 1 and activator_fr == 1) or delay_rouge >= 0):
                    delay_rouge = delay_rouge - 1
                    cv2.putText(frame,'STOP', bottomRightCornerOfText, font, fontScale,(0, 0, 255),lineType)                 
                
                cv2.imshow("Processed", frame)
                
                

                
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
            
        cap.release()
        cv2.destroyAllWindows()
else:
    for img_path in glob.glob('feux vert\*.jpg'):
        frame = cv2.imread(img_path)


# Appeler fonctions ici ---------------------------------------------
# Argument = frame
        frame= resize(frame)
        frame = feu_vert(frame)
        frame = feu_rouge(frame)
                
                
        cv2.imshow("Processed", frame)
        

        cv2.waitKey(0)
        cv2.destroyAllWindows()



    
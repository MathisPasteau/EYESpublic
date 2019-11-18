# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:31:26 2019

@author: 33672
"""


import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.logger import Logger

import numpy as np 
import cv2 
import time

class myapp(BoxLayout):
    def __init__(self,**kwargs):
        super(myapp,self).__init__(**kwargs)
        self.padding = 250

        btn1 = Button(text='Close App')
        btn1.bind(on_press=self.clkfunc)
        self.add_widget(btn1)
        
        btn2 = Button(text='Passage Piétons')
        btn2.bind(on_press=self.passagepietons)
        self.add_widget(btn2)

    def clkfunc(self , obj):
        App.get_running_app().stop()
        Window.close()
        
    def passagepietons(self , obj):
        #cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture("videotestcpt3.mp4")
        
        width = 400
        height = 280
        dim = (width, height)
        
        # WHITE color
        low_red = np.array([100, 100, 100])
        high_red = np.array([255, 255, 255])
        
        # Write some Text
        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (30,250)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        
        wait = 0
        secure = 0
        seconde = 0
        
        
        
        def region_of_interest(thresh_image):
            hauteur = thresh_image.shape[0]
            largeur = thresh_image.shape[1]
            mask = np.zeros_like(thresh_image)
            #4 points 1:(N,O) 2:(S,O) 3:(N,E) 4:(S,E)
            rectangle = np.array([[[
            (0, 80),
            (0, hauteur),
            (largeur, hauteur),
            (largeur, 80),
            ]]], np.int32)
        
            cv2.fillPoly(mask, rectangle, 255)
            image_masquee = cv2.bitwise_and(thresh_image, mask)
            return image_masquee
        
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        
        while(cap.isOpened()):    
            
            count = 0
            wait = wait - 1
            second = (seconde + 1)%60
            
            _, frame = cap.read()   
            
            
            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite("origin.jpg", frame)  
            hsv_frame = frame
             
            red_mask = cv2.inRange(hsv_frame, low_red, high_red)
            img = cv2.bitwise_and(frame, frame, mask=red_mask)    
            
            cv2.imwrite("frame.jpg", img) 
            # Reading image 
            img2 = cv2.imread('origin.jpg', cv2.IMREAD_COLOR) 
               
            # Reading same image in another variable and  
            # converting to gray scale. 
            img = cv2.imread('frame.jpg', cv2.IMREAD_GRAYSCALE)
            
            img = region_of_interest(img)
               
            # Converting image to a binary image  
            # (black and white only image). 
            _,threshold = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
            cv2.imshow('threshold', threshold)
               
            # Detecting shapes in image by selecting region  
            # with same colors or intensity. 
            contours,_=cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
               
            # Searching through every region selected to  
            # find the required polygon. 
            for cnt in contours : 
                area = cv2.contourArea(cnt) 
               
                # Shortlisting the regions based on there area. 
                if (area > 400 and area < 5000):  
                    approx = cv2.approxPolyDP(cnt,  
                                              0.01 * cv2.arcLength(cnt, True), True) 
               
                    # Checking if the no. of sides of the selected region is 7. 
                    if(len(approx) == 4): 
                        cv2.drawContours(img2, [approx], 0, (255, 255, 255), 2)
                        count = count + 1
            
            if(count >= 3):
                wait = 60
                secure = secure + 1
                seconde = 1
            if(seconde == 59):
                secure = 0
            if((count >= 3 or wait >= 0) and secure >= 6 ):
                cv2.putText(img2,'PASSAGE PIETON', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
                    
            # Showing the image along with outlined arrow. 
            cv2.imshow('image2', img2)    
        
            # Exiting the window if 'q' is pressed on the keyboard. 
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
               
            
        
        cap.release()
        cv2.destroyAllWindows()

class SimpleKivy(App):
    def on_stop(self):
        Logger.critical('App: Aaaargh I\'m dying!')
        
    def build(self):
        return myapp()

if __name__ == '__main__':
    SimpleKivy().run()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:46:03 2020

@author: franciscorealescastro
"""

import cv2
import numpy as np
from scipy import ndimage
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detectarPlaca(img):
    I=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    u,_=cv2.threshold(I,0,255,cv2.THRESH_OTSU)
    mascara = np.uint8(255*(I>u))
    output=cv2.connectedComponentsWithStats(mascara,4,cv2.CV_32S)
    cantObj=output[0]
    labels=output[1]
    stats=output[2]
    maskObj=[]
    maskConv=[]
    diferenciaArea=[]
    for i in range(1,cantObj):
        if stats[i,4]>stats[:,4].mean():
            mascara=ndimage.binary_fill_holes(labels==i)
            mascara=np.uint8(255*mascara)
            maskObj.append(mascara)
            #calculo convexhull
            contours,_=cv2.findContours(mascara,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnt=contours[0]
            hull=cv2.convexHull(cnt)
            puntosConvex=hull[:,0,:]
            m,n=mascara.shape
            ar=np.zeros((m,n))
            mascaraCovex=np.uint8(255*cv2.fillConvexPoly(ar,puntosConvex,1))
            maskConv.append(mascaraCovex)
            #comparacion area CH con objeto
            areaObj=np.sum(mascara)/255
            areaConv=np.sum(mascaraCovex)/255
            diferenciaArea.append(np.abs(areaObj-areaConv))
            
    maskPlaca=maskConv[np.argmin(diferenciaArea)]   
   
    # correccion perspectiva
    
    
    vertices=cv2.goodFeaturesToTrack(maskPlaca,4,0.01,10)
    x=vertices[:,0,0]
    y=vertices[:,0,1]
    vertices=vertices[:,0,:]
    xo=np.sort(x)
    yo=np.sort(y)
    
    xn=np.zeros((1,4))
    yn=np.zeros((1,4))
    n=(np.max(xo)-np.min(xo))
    m=(np.max(yo)-np.min(yo))
    
    xn=(x==xo[2])*n+(x==xo[3])*n
    
    yn=(y==yo[2])*m+(y==yo[3])*m
    verticesN=np.zeros((4,2))
    verticesN[:,0]=xn
    verticesN[:,1]=yn
    
    vertices=np.int64(vertices)
    verticesN=np.int64(verticesN)
    
    h,_=cv2.findHomography(vertices,verticesN)
    
    placa=cv2.warpPerspective(img,h,(np.max(verticesN[:,0]),(np.max(verticesN[:,1]))))



    text = pytesseract.image_to_string(placa,config='--psm 11')

    return text


#img=cv2.imread("car6.jpg")

cap=cv2.VideoCapture('video.mp4')

while (cap.isOpened()):
    
    ret,frame=cap.read()
    
    if ret==True:
        imgresize=cv2.resize(frame,(1200,900))
        text=detectarPlaca(imgresize)
        print(text)
        cv2.putText(imgresize,"La placa es: "+ text,(10,300),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,255,255),1)
        cv2.imshow("placa", imgresize)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.waitKey(0)
cv2.destroyAllWindows()

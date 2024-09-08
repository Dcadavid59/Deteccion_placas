# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 20:27:51 2024

@author: david
"""

import cv2
import numpy as np
from scipy import ndimage

def detectarPlaca(img):
    imgresize=cv2.resize(img,(800,600))
    alto,ancho,_=imgresize.shape
    xL=int(ancho/9)
    xR=xL*8
    yA=int(alto/2)-50
    yB=yA*2
    
    # Dibujo del rectangulo donde se extraera las placas
    cv2.rectangle(imgresize,(xR,yA),(xL,yB),(0,255,0),2)
    
    # Recorte de zona de interes
    recorte=imgresize[yA:yB,xL:xR]
        
    # Elegimos el umbral de verde en HSV
    umbral_bajo = np.array([10,100,50], np.uint8)
    umbral_alto = np.array([40,255,255], np.uint8)
    
    recorte = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(recorte, cv2.COLOR_RGB2HSV)
    
    # hacemos la mask y filtramos en la original
    mask = cv2.inRange(img_hsv, umbral_bajo, umbral_alto)
    
    output=cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)
    cantObj=output[0]
    labels=output[1]
    stats=output[2]
    maskObj=[]
    maskConv=[]
    diferenciaArea=[]
    
    for i in range(1,cantObj):
        if stats[i,4]>stats[:,4].mean():
            mask2=ndimage.binary_fill_holes(labels==i)
            mask2=np.uint8(255*mask2)
            maskObj.append(mask2)
            #calculo convexhull
            _contours,_=cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt=_contours[0]
            hull=cv2.convexHull(cnt)
            puntosConvex=hull[:,0,:]
            m,n=mask2.shape
            ar=np.zeros((m,n))
            mascaraCovex=np.uint8(255*cv2.fillConvexPoly(ar,puntosConvex,1))
            maskConv.append(mascaraCovex)
            #comparacion area CH con objeto
            areaObj=np.sum(mask2)/255
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
    placa=cv2.warpPerspective(recorte,h,(np.max(verticesN[:,0]),(np.max(verticesN[:,1]))))
    return placa

"""
img=cv2.imread("carro_suzuki.jpg")
placa = detectarPlaca(img)
cv2.imshow('',placa)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

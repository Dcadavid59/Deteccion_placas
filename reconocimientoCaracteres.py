#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 20:59:49 2020

@author: franciscorealescastro
"""
import cv2
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


def get_hog():
    winSize = (20,20)
    blockSize=(8,8)
    blockStride = (4,4)
    cellSize=(8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 2.
    histrogramType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlavels = 64
    signedGradient = True 
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histrogramType,L2HysThreshold,gammaCorrection,nlavels,signedGradient)
    return hog

def escalar(img,m,n):
    if m>n:
        imgN=np.uint8(255*np.ones((m,round((m-n)/2),3)))
        escalada=np.concatenate((np.concatenate((imgN,img),axis=1),imgN), axis=1)
    else:
        imgN=np.uint8(255*np.ones((round((n-m)/2),n,3)))
        escalada=np.concatenate((np.concatenate((imgN,img),axis=0),imgN), axis=0)
    
    img = cv2.resize(escalada, (20,20))
    return img
        
def obtenerDatos():
    posiblesEtiq=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
    datos = []
    etiquetas = []
    
    for i in range(1,20):
        for j in posiblesEtiq:
            img=cv2.imread(j+'-'+str(i)+".jpg")
            if img is not None:
                m,n,_=img.shape
                if m !=20 or n !=20:
                    img =escalar(img,m,n)
                etiquetas.append(np.where(np.array(posiblesEtiq)==j)[0][0])
                hog = get_hog()
            datos.append(np.array(hog.compute(img)))
    datos=np.array(datos)[:,:]
    #datos=np.array(datos)[:,:,0]
    etiquetas=np.array(etiquetas)
    return datos, etiquetas

def clasificadorCaracteres():
    datos, etiquetas=obtenerDatos()
    knn=KNeighborsClassifier(n_neighbors=1)
    knn.fit(datos,etiquetas)
    SVM=svm.SVC(kernel='linear', probability=True, random_state=0,gamma='auto')
    SVM.fit(datos,etiquetas)
    return knn, SVM

#knn,SVM=clasificadorCaracteres()
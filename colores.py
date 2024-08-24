# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:20:46 2024

@author: david
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("vw_amarilla3.jpg")

# Elegimos el umbral de verde en HSV
umbral_bajo = np.array([10,100,50], np.uint8)
umbral_alto = np.array([40,255,255], np.uint8)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# hacemos la mask y filtramos en la original
mask = cv2.inRange(img_hsv, umbral_bajo, umbral_alto)

_contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, _contours, -1, (255,255,0), 3)
    
cv2.imshow('',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
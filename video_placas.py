import numpy as np
import pytesseract
import cv2
from PIL import Image
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

cap=cv2.VideoCapture('video_moto2.mp4')

cont=0

while (cap.isOpened()):
    
    ret,frame=cap.read()
    
    if ret==True:
        
        al,an,_=frame.shape

        #print(f'alto: {al} \n ancho: {an}')
        
    
        x1=int(an/3) #se toma 1/3 de la imagen
        x2=int(x1*2) #hasta el inicio de 3/3 de la imagen
    
        y1=int(al/3)
        y2=int(al*2)

        imgresize=cv2.resize(frame,(1200,900))
        gray = cv2.cvtColor(imgresize, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5),0)
        canny = cv2.Canny(gray,100,200)
        cnts,_=cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        
        #cv2.rectangle(canny,(x1,y1),(x2,y2),(0,255,0),2)
        for i in cnts:
            area=cv2.contourArea(i)
            x,y,w,h = cv2.boundingRect(i)
            epsilon = 0.01*cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,epsilon,True)
    
            if (len(approx)==4) and (area > 6000):
                print(f'aprox : {len(approx)}')
                print(f'area={area}')
                cv2.drawContours(imgresize,[i],0,(119, 255, 51),2)
                aspect_ratio = float(w)/h
                print(f'aspect ratio: {aspect_ratio}')
                if aspect_ratio>1:
                    
                    placa = gray[y:y+h,x:x+w]
            
                    text = pytesseract.image_to_string(placa,config='--psm 11')
            
                #cv2.imshow('PLACA',placa)
                #cv2.moveWindow('PLACA',10,50)
                cv2.rectangle(imgresize,(x,y),(x+w,y+h),(119, 255, 51),3)
                cv2.putText(imgresize,text,(x-40,y-20),1,2.2,(0,255,0),2,cv2.LINE_AA)

        cv2.imshow('video_moto2',imgresize)
        
        
        
        
        if (cv2.waitKey(1)==ord('s')):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
        
    
    # #rectangulo donde mostrara la placa
    # cv2.rectangle(frame,(870,750),(1070,850),(0,0,0),cv2.FILLED)
    # cv2.putText(frame,ctexto[0:7],(900,810),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
    # al,an,_=frame.shape
    
    # x1=int(an/3) #se toma 1/3 de la imagen
    # x2=int(x1*2) #hasta el inicio de 3/3 de la imagen
    
    # y1=int(al/3)
    # y2=int(al*2)
    
    # #rectangulo de procesar placa
    # cv2.rectangle(frame,(x1+160,y1+500),(1120,940),(0,0,0),cv2.FILLED)
    # cv2.putText(frame,'procesando placa',(x1+180,y1+550),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
    #zona donde se extraera las placas
    #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    
    # #obteniendo recorte de zona de interes
    # recorte=frame[y1:y2,x1:x2]
    
    # nazul=np.matrix(recorte[:,:,0])
    # nverde=np.matrix(recorte[:,:,1])
    # nrojo=np.matrix(recorte[:,:,2])
    
    # color_amarillo=cv2.absdiff(nverde,nazul)
    
    
    
    
    

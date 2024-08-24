import numpy as np
import pytesseract
import cv2
from PIL import Image
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


cap=cv2.VideoCapture('video_moto2.mp4')

ctexto=''


while (cap.isOpened()):
    
    ret,frame=cap.read()
    
    if ret==True:
        
        imgresize=cv2.resize(frame,(1000,900))
        
        # #rectangulo donde mostrara la placa
        
        cv2.rectangle(imgresize,(300,600),(500,800),(0,0,0),cv2.FILLED)
        cv2.putText(imgresize,ctexto[0:7],(300,650),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        al,an,_=imgresize.shape
        
        
        x1=int(an/3) #se toma 1/3 de la imagen
        x2=int(x1*2) #hasta el inicio de 3/3 de la imagen
    
        y1=int(al/3)
        y2=int(y1*2)
        
        #rectangulo de procesar placa
        # cv2.rectangle(frame,(x1+160,y1+500),(1120,940),(0,0,0),cv2.FILLED)
        # cv2.putText(frame,'procesando placa',(x1+180,y1+550),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        #zona donde se extraera las placas
        
        cv2.rectangle(imgresize,(x1,y1),(x2,y2),(0,255,0),2)
        
        #obteniendo recorte de zona de interes
        
        recorte=imgresize[y1:y2,x1:x2]
        
        nazul=np.matrix(recorte[:,:,0])
        nverde=np.matrix(recorte[:,:,1])
        nrojo=np.matrix(recorte[:,:,2])
    
        color_amarillo=cv2.absdiff(nverde,nazul)
        
        #binarizamos la imagen
        
        _,umbral=cv2.threshold(color_amarillo,40,255,cv2.THRESH_BINARY)
        
        #cany = canny = cv2.Canny(umbral,100,200)
        
        #buscamos los contornos
        
        contornos,_= cv2.findContours(umbral,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        #se ordenan los contornos de mayor a menor
        
        contornos= sorted(contornos,key=lambda x: cv2.contourArea(x),reverse=True)
        
        for contorno in contornos:
            
            area=cv2.contourArea(contorno)
            
            if area > 500 and area < 7000:
                
                #detectamos la placa
                x,y,ancho,alto=cv2.boundingRect(contorno)
                
                #extraemos cordenadas
                
                xpi= x + x1   #cordenada de la placa en x inicial
                ypi= y + y1   #cordenada de la placa en y inicial
                
                xpf= x + ancho + x1 #cordenada de la placa en x final
                ypf= y + alto + y1  #cordenada de la placa en y final
                
                #dibujamos el rectangulo
                
                cv2.rectangle(imgresize,(xpi,ypi),(xpf,ypf),(255,255,0),2)
                
                #extraemos los pixeles
                
                placa= imgresize[ypi:ypf,xpi:xpf]
                
                
                #extraemos el alto y ancho de fotogramas
                alp,anp,cp=placa.shape
                
                
                
                #procesamos los pixeles para extraer los valores de las placas
                
                Mva = np.zeros((alp,anp))
                
                #normalizamos las matrices
                
                nblue= np.matrix(placa[:,:,0])
                ngreen= np.matrix(placa[:,:,1])
                nred= np.matrix(placa[:,:,2])
                
                #se crea una mascara
                
                for col in range(0,alp):
                    for fil in range(0,anp):
                        Max= max(nred[col,fil],ngreen[col,fil],nblue[col,fil])
                        Mva[col,fil] = 255 - Max
                        
                #binarizamos la imagen
                _, bin = cv2.threshold(Mva,150,255,cv2.THRESH_BINARY)
                
                #convertimos la matriz en imagen
                bin = bin.reshape(alp,anp)
                
                bin = Image.fromarray(bin)
                
                bin = bin.convert("L")
                
                # nos aseguramos de tener un buen tamaño de placa
            
                if alp >= 10 and anp >= 40:

                    config_placa = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    
                    texto = pytesseract.image_to_string(placa,config= config_placa)
                    
                
                    
                    #condicion para no mostrar basura
                    
                    if len(texto) >= 7:
                    
                    #texto = pytesseract.image_to_string(placa,config='--psm 11')
                        
                        print(texto)
                        
                        ctexto=texto
                
                break
                
    
        
        cv2.imshow('video',imgresize)
        
        
        if (cv2.waitKey(1)==ord('s')):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()
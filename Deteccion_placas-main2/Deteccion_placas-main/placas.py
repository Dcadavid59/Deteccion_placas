import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
placa = []

image = cv2.imread('placa1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray,(4,4))
canny = cv2.Canny(gray,250,250)
cnts,_=cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image,cnts,-1,(0,255,0),2)

for i in cnts:
    area=cv2.contourArea(i)
    x,y,w,h = cv2.boundingRect(i)
    epsilon = 0.09*cv2.arcLength(i,True)
    approx = cv2.approxPolyDP(i,epsilon,True)
    print(len(approx))

    if len(approx)==4 and area > 6000:
        print(f'area={area}')
        cv2.drawContours(image,[i],0,(119, 255, 51),2)
        
        aspect_ratio = float(w)/h
        print(aspect_ratio)
        if aspect_ratio>1.9:
            placa = gray[y:y+h,x:x+w]
            print(f'placa1 : {placa}')
            text = pytesseract.image_to_string(placa,config='--psm 11')
            print('PLACA: ',text)
            cv2.imshow('PLACA',placa)
            cv2.moveWindow('PLACA',10,50)
            cv2.rectangle(image,(x,y),(x+w,y+h),(119, 255, 51),3)
            cv2.putText(image,text,(x-40,y-20),1,2.2,(0,255,0),2,cv2.LINE_AA)

cv2.imshow('Image2',image)
#cv2.moveWindow('Image',45,10)
cv2.waitKey(0)
cv2.destroyAllWindows()

# gray = cv2.blur(gray,(3,3))
# canny = cv2.Canny(gray,150,200)
# canny = cv2.dilate(canny,None,iterations=1)

# cnts,_ = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(image,cnts,-1,(0,255,0),2)











# _,cnts,_ = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# cnts,_ = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# #cv2.drawContours(image,cnts,-1,(0,255,0),2)
# for c in cnts:
#   area = cv2.contourArea(c)
#   x,y,w,h = cv2.boundingRect(c)
#   epsilon = 0.09*cv2.arcLength(c,True)
#   approx = cv2.approxPolyDP(c,epsilon,True)
  
#   if len(approx)==4 and area>9000:
#     print('area=',area)
#     #cv2.drawContours(image,[approx],0,(0,255,0),3)
#     aspect_ratio = float(w)/h
#     if aspect_ratio>2.4:
#       placa = gray[y:y+h,x:x+w]
#       text = pytesseract.image_to_string(placa,config='--psm 11')
#       print('PLACA: ',text)
#       cv2.imshow('PLACA',placa)
#       cv2.moveWindow('PLACA',780,10)
#       cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
#       cv2.putText(image,text,(x-20,y-10),1,2.2,(0,255,0),3)
      
# cv2.imshow('Image',image)
# cv2.moveWindow('Image',45,10)
# cv2.waitKey(0)
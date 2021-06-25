from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import cv2 
import numpy as np
import os
import tensorflow as tf
from PIL import Image, ImageTk
  

main = tkinter.Tk()
main.title("Deep HDR Reconstruction of Dynamic Scenes") 
main.geometry("600x500")

global filename
global loaded_model
img_list = []
files = []
def upload(): 
    global filename
    files.clear()
    filename = filedialog.askdirectory(initialdir="images")
    index = 0
    for root, dirs, file in os.walk(filename):
        for fdata in file:
            files.append(root+"/"+fdata)
            img = cv2.imread(root+"/"+fdata)
            img = cv2.resize(img,(400,400))
            cv2.imshow("input image "+str(index), img)
            index = index + 1
    cv2.waitKey();            
            
    

def Alignment():
    print(files)
    #img_list.clear()
    img_list = [cv2.imread(fn) for fn in files]
    for i in range(len(img_list)):
        img_list[i] = cv2.resize(img_list[i],(400,400),fx=0,fy=0, interpolation=cv2.INTER_NEAREST)
    print('aligning the images..................................................................................................')
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(img_list, img_list)
    print('merging the images....................................................................................................') 
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(img_list)
    res_mertens_8bit = np.clip(res_mertens*255, 3, 255).astype('uint8')
    cv2.imwrite("aligned.jpg", res_mertens_8bit)
    
    print('upscaling the image........................................................................................please wait..')
    
    #ldr_img = cv2.imread("test.jpg",3)
    img=cv2.imread("aligned.jpg")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "ESPCN_x4.pb"
    sr.readModel(path)
    sr.setModel("espcn", 4) 
    result = sr.upsample(img) 
    cv2.imwrite("us.jpg", result)
    #tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
    #img=cv2.imread("output.jpg")
    #ldrMantiuk = tonemapMantiuk.process(img)
    #ldrMantiuk = 3 * ldrMantiuk
    #cv2.imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255)
    #ldr_img = cv2.resize(ldr_img,(400,400))
    #tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    #img=cv2.imread("output.jpg")
    #ldrDrago = tonemapDrago.process(img)
    #ldrDrago = 3 * ldrDrago
    #cv2.imwrite("ldr-Drago.jpg", ldrDrago * 255)
    
  #  index = 0
  #  for i in range(len(files)):
   #     img = cv2.imread(files[i])
    #    img = cv2.resize(img,(400,400))
     #   cv2.imshow("input"+str(index), img)
      #  index = index + 1
  #  test=cv2.imread("test.jpg")
   # tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
   # ldrMantiuk = tonemapMantiuk.process(test)
   # ldrMantiuk = 3 * ldrMantiuk
   # cv2.imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255)
    #cv2.waitKey();
    print('toning the image...................................................................................................')
    img = cv2.imread("us.jpg", 1)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,30)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('final.jpg', img)
    img=cv2.imread('final.jpg')
    img=cv2.resize(img,(400,400))
    cv2.imshow('final',img)
    print('image saved')
        

def exit():
    global main
    main.destroy()
    
  
font = ('times', 16, 'bold')
title = Label(main, text='Deep HDR Reconstruction of Dynamic Scenes', justify=LEFT)
title.config(bg='white', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()
image = Image.open("bg1.jpg")
resize_image = image.resize((600, 450))
img = ImageTk.PhotoImage(resize_image)

label = Label(
    main,
    image=img
)
label.place(x=0, y=70)

font1 = ('times', 14, 'bold')
uploadimages = Button(main, text="Upload Different Exposure Images", command=upload)
uploadimages.place(x=200,y=100)
uploadimages.config(font=font1)  

LDRbutton = Button(main, text="Run", command=Alignment)
LDRbutton.place(x=200,y=150)
LDRbutton.config(font=font1) 


exitapp = Button(main, text="Exit", command=exit)
exitapp.place(x=200,y=250)
exitapp.config(font=font1) 

main.config(bg='blue')
main.mainloop()

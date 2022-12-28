from PIL import Image, ImageDraw
import tkinter as tk
import numpy as np
import cv2
from sklearn.svm import SVC
import pandas as pd
import pickle 
from matplotlib.image import imread

traindf = pd.read_csv(r'D:/python/ml/mnist_train.csv')



x_train = traindf.drop('label',axis=1)
y_train = traindf['label']
svm_rbf = SVC(kernel="rbf")
svm_rbf.fit(x_train, y_train)

'''
filename = 'svm_model'
pickle.dump(svm_rbf, open(filename, 'wb'))
model = pickle.load(open(filename, 'rb'))

'''

root = tk.Tk()
root.resizable('0', '0')

canvas = tk.Canvas(root, bg='black', height=400, width=400)
canvas.grid(row=0, column=0, columnspan=4)
img = Image.new('RGB', (400,400), ('black'))
imagedraw = ImageDraw.Draw(img)
count = 0

def draw(event):
    x , y = event.x , event.y
    x1 , y1 = x-20 , y-20
    x2, y2 = x+20 , y+20

    canvas.create_oval((x1, y1,x2,y2), fill='white', outline='white')
    imagedraw.ellipse((x1, y1,x2,y2), fill='white', outline='white')

def clear_canvas():
    global img, imagedraw
    img = Image.new('RGB', (400,400), ('black'))
    imagedraw = ImageDraw.Draw(img)
    canvas.delete('all')
    return

def predict():
    global count
    imagearray = np.array(img)
    imagearray = cv2.cvtColor(imagearray, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(imagearray, (28,28))

    cv2.imwrite(str(count)+'.jpg',image1)
    read_img = imread(str(count)+'.jpg')
    convert = np.array(read_img).reshape(-1,784)

    count = count+1
    
    pred_svm_rbf = svm_rbf.predict(convert)
    print(pred_svm_rbf)
    
    
    

canvas.bind("<B1-Motion>", draw)

button_predict = tk.Button(root, text='PREDICT', width=15, height=2, bg='black', fg='white', font='Helvetica', command= predict)
button_predict.grid(row=2,column=0)

button_clear = tk.Button(root, text='CLEAR', width=15, height=2, bg='black', fg='white', font='Helvetica', command= clear_canvas)
button_clear.grid(row=2,column=2)

button_exit= tk.Button(root, text='EXIT', width=15, height=2, bg='black', fg='white', font='Helvetica', command= root.destroy)
button_exit.grid(row=2,column=3)

root.mainloop()
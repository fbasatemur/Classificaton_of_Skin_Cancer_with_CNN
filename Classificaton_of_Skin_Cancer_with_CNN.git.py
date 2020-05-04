# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:41:38 2020

@author: fbasatemur
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image

import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np

from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import load_model
from keras.optimizers import Adam

#%%

skinDF = pd.read_csv("HAM10000_metadata.csv")

skinDF.info()

#%% preprocessing

dataFolderName = "./mnist_ham10000_dataset/"
imageExtansion = ".jpg"

skinDF["path"] = [ dataFolderName + i + imageExtansion for i in skinDF["image_id"]] 
skinDF["image"] = skinDF["path"].map(lambda x: np.asarray(Image.open(x).resize((100,75))))

plt.imshow(skinDF["image"][0])

skinDF["dx_id"] = pd.Categorical(skinDF["dx"]).codes
skinDF.to_pickle("skinDF.pkl")

#%% load pkl

skinDF = pd.read_pickle("skin_df.pkl")

#%% standartization

x_train = np.asarray(skinDF["image"].to_list())
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_train = (x_train - x_train_mean)/x_train_std

# one hot encoding
y_train = to_categorical(skinDF["dx_id"], num_classes = 7)

#%% CNN

inputShape = (75,100,3)
numClasses = 7

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation = "relu", padding = "Same", input_shape = inputShape))
model.add(Conv2D(32, kernel_size=(3,3), activation = "relu", padding = "Same"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3,3), activation = "relu", padding = "Same"))
model.add(Conv2D(64, kernel_size=(3,3), activation = "relu", padding = "Same"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(numClasses, activation="softmax"))
model.summary()

optimizer = Adam(lr = 0.0001)
model.compile(optimizer=optimizer, loss= "categorical_crossentropy", metrics = ["accuracy"])

epochs = 5
batchSize = 25

hitory = model.fit(x = x_train, y= y_train, batch_size = batchSize, epochs = epochs, verbose = 1, shuffle = True)

model.save("model_save2.h5")

#%% model load

model1 = load_model("model_save1.h5")
model2 = load_model("model_save2.h5")


#%% GUI

window = tk.Tk()
window.geometry("1080x640")
window.title("Skin Cancer Classification")


# frames

frameLeft = tk.Frame(window, width = 540, height = 640, bd = "2")
frameLeft.grid(row=0, column=0);

frameRight = tk.Frame(window, width = 540, height = 640, bd = "2")
frameRight.grid(row=0, column=1);

frame1 = tk.LabelFrame(frameLeft, text = "Image", width = 530, height = 500)
frame1.grid(row=1, column = 0)

frame2 = tk.LabelFrame(frameLeft, text = "Model & Save", width = 530, height = 130)
frame2.grid(row=0, column = 0)

frame3 = tk.LabelFrame(frameRight, text = "Features", width = 260, height = 630)
frame3.grid(row=0, column = 0)

frame4 = tk.LabelFrame(frameRight, text = "Result", width = 260, height = 630)
frame4.grid(row=0, column = 1, padx = 10)


# frame1

def imageResize(img):
      basewidth = 500
      wpercent = (basewidth/float(img.size[0]))
      hsize = int(float(img.size[1])*float(wpercent))
      img = img.resize((basewidth, hsize), Image.ANTIALIAS)
      return img


imageName = ""
imagePath = ""

def openImage():

      global imageName 
      global imagePath 
      inputFlag = False
      
      
      if inputFlag:
            messagebox.showinfo(title = "Warning", message="Only one image can be opened !")
      else:
            inputFlag = True
            imagePath = filedialog.askopenfilename(initialdir= "./", title= "Select an image file")
            imageName = imagePath.split("/")[-1].split(".")[0]
            
            # image label
            tk.Label(frame1, text = imageName, bd = 3).pack(pady = 10)
            
            # imshow
            image = Image.open(imagePath)
            image = imageResize(image)
            image = ImageTk.PhotoImage(image)
            panel = tk.Label(frame1, image = image)
            panel.image = image
            panel.pack(padx = 15, pady = 10)
            
            # image features
            data = pd.read_csv("HAM10000_metadata.csv")
            cancer = data[data.image_id == imageName]
            
            for i in range(cancer.size):
                  x = 0.5
                  y = ((i+1)/20)
                  tk.Label(frame3, font = ("Ubuntu", 10), text = str(cancer.iloc[0,i])).place(relx = x, rely = y)
 
           
menubar = tk.Menu(window)
window.config(menu = menubar)

file = tk.Menu(menubar)
menubar.add_cascade(label = "File", menu = file)
file.add_command(label = "Open", command = openImage)


# frame3

def classification():
      if imagePath != "" and  models.get() != "":
            
            if models.get() == "Model1":
                  classificationModel = model1
            else:
                  classificationModel = model2
            
            z = skinDF[skinDF.image_id == imageName]
            z = z.image.values[0].reshape(1,75,100,3)
            
            # standartization
            z = (z - x_train_mean) / x_train_std
            h = classificationModel.predict(z)[0]
            h_index = np.argmax(h)
            predictedCancer = list(skinDF.dx.unique())[h_index]
            
            for i in range(len(h)):
                  x = 0.5
                  y = ((i+1)/20)
                  
                  if i != h_index:
                        tk.Label(frame4, text = str(h[i])).place(relx = x, rely = y)
                  else:
                        tk.Label(frame4, bg = "green", fg = "white", text = str(h[i])).place(relx = x, rely = y)
                        
            if chvar.get() == 1:
                  val = entry.get()
                  entry.config(state = "disabled")
                  path_name = val + ".txt"
                  
                  save_txt = imagePath + " --> " + str(predictedCancer)
                  
                  textFile = open(path_name, "w")
                  textFile.write(save_txt)
                  textFile.close()
      else:
            messagebox.showinfo(title = "Warning", message = "Choose image and model first !")
            tk.Label(frame3, text = "Choose image and model first !").place(relx = 0.1, rely = 0.6)
      

columns = ["lessions_id", "image_id", "dx", "dx_type", "age", "sex", "localization"]

for i in range(len(columns)):
      x = 0.1
      y = ((i+1)/20)
      tk.Label(frame3, font = ("Ubuntu", 10), text = str(columns[i]) + ": ").place(relx = x, rely = y)

classifyButton = tk.Button(frame3, bg = "white", fg = "black", bd = 4, font = ("Ubuntu",10), 
                           activebackground = "green", activeforeground = "white", text = "Classify", command = classification)

classifyButton.place(relx = 0.1, rely = 0.5)


# frame 4

labels = skinDF.dx.unique()

for i in range(len(columns)):
      x= 0.1
      y = ((i+1)/20)
      tk.Label(frame4, font = ("Ubuntu", 10), text = str(labels[i]) + ": ").place(relx = x, rely = y)



# frame 2
 # combo box

modelSelectionLabel = tk.Label(frame2, text = "Choose classsification model:")
modelSelectionLabel.grid(row = 0, column = 0, padx = 5)

models = tk.StringVar()
modelSelection = ttk.Combobox(frame2, textvariable = models, values = ("Model 1", "Model2" ), state = "readonly")
modelSelection.grid(row = 0, column = 1, padx = 5)

 # checkBox
chvar = tk.IntVar()
chvar.set(0)
xbox = tk.Checkbutton(frame2, text = "Save Classification Result", variable = chvar)
xbox.grid(row = 1, column = 0, pady = 5)

 # entry
entry = tk.Entry(frame2, width = 23)
entry.insert(string = "Saving name ..", index = 0)
entry.grid(row = 1, column = 1)



window.mainloop()


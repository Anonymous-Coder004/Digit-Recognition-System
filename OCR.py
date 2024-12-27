import tensorflow
import numpy as np
import os
import cv2 as cv
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
img=[]
testratio=0.2
valratio=0.2
imageDimensions=(32,32,3)
classNo=[]
path='myData' 
myList=os.listdir(path)
print(len(myList))
n_o_classes=len(myList)
for i in range(0,n_o_classes):
    myPic_list=os.listdir(path+"/"+str(i))
    for x in myPic_list:
        curImg=cv.imread(path+"/"+str(i)+"/"+x)
        curImg=cv.resize(curImg,(imageDimensions[0],imageDimensions[1]))
        img.append(curImg)
        classNo.append(i)
    print(i,end=" ")
img=np.array(img)
classNo=np.array(classNo)
X_train,X_test,y_train,y_test=train_test_split(img,classNo,test_size=testratio)
X_train,X_validation,y_train,y_validation=train_test_split(X_train,y_train,test_size=valratio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)
n_of_samples=[]
for i in range(0,n_o_classes):
    n_of_samples.append(len(np.where(y_train==i)[0]))
plt.figure(figsize=(10,5))
plt.bar(range(0,n_o_classes),n_of_samples)
plt.title("No of images for each class")
plt.xlabel("Class ID")
plt.ylabel("No of images")
plt.show()
def preProcessing(img):
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img=cv.equalizeHist(img)#enhance the contrast so that text can be read easily
    img=img/255
    return img
X_train=np.array(list(map(preProcessing,X_train)))
X_test=np.array(list(map(preProcessing,X_test)))
X_validation=np.array(list(map(preProcessing,X_validation)))
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
dataGen=ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1,rotation_range=10)
dataGen.fit(X_train)
y_train=to_categorical(y_train,n_o_classes) #for one hot encoding
y_test=to_categorical(y_test,n_o_classes)
y_validation=to_categorical(y_validation,n_o_classes)
def myModel():
    noOfFilters=60
    sizeOfFilter1=(5,5)
    sizeOfFilter2=(3,3)
    sizeOfPool=(2,2)
    noOfNodes=500
    model=Sequential()
    model.add((tensorflow.keras.layers.Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imageDimensions[0],imageDimensions[1],1),activation='relu')))
    model.add((tensorflow.keras.layers.Conv2D(noOfFilters,sizeOfFilter1,activation='relu')))
    model.add((tensorflow.keras.layers.MaxPooling2D(pool_size=sizeOfPool)))
    model.add((tensorflow.keras.layers.Conv2D(noOfFilters//2,sizeOfFilter2,activation='relu')))
    model.add((tensorflow.keras.layers.Conv2D(noOfFilters//2,sizeOfFilter2,activation='relu')))
    model.add((tensorflow.keras.layers.MaxPooling2D(pool_size=sizeOfPool)))
    model.add(tensorflow.keras.layers.Flatten())
    model.add(tensorflow.keras.layers.Dense(noOfNodes,activation='relu'))
    model.add(tensorflow.keras.layers.Dropout(0.5))
    model.add(tensorflow.keras.layers.Dense(n_o_classes,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
model=myModel()
print(model.summary())
batch_size_val=50
epochs_val=10
steps_per_epoch_val=2000                                                                    
history=model.fit_generator(dataGen.flow(X_train,y_train,
                                 batch_size=batch_size_val),
                                 epochs=epochs_val,
                                 validation_data=(X_validation,y_validation),
                                 shuffle=1)
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.show()
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
score=model.evaluate(X_test,y_test,verbose=0)
print('Test Score=',score[0])
print('Test Accuracy=',score[1])
pickle_out=open("model_trained.pkl","wb")
pkl.dump(model,pickle_out)
pickle_out.close()



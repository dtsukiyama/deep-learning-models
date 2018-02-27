import numpy as np
import math
import cv2                     
from PIL import Image
import time 
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam


class faceDetection(object):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
        
    def faces(self, image):
        # Convert the RGB  image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect the faces in image
        faces = self.face_cascade.detectMultiScale(gray, 4, 6)

        # Print the number of faces detected in the image
        print('Number of faces detected:', len(faces))

        # Make a copy of the orginal image to draw face detections on
        image_with_detections = np.copy(image)

        # Get the bounding box for each detected face
        for (x,y,w,h) in faces:
            # Add a red bounding box to the detections image
            cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)
            
        return image_with_detections

    def denoisefaces(self, image):
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 20, 15, 7, 21)

        # Convert the RGB  image to grayscale
        gray_noise = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2GRAY)

        # Detect the faces in image
        faces = self.face_cascade.detectMultiScale(gray_noise, 4, 6)

        # Print the number of faces detected in the image
        print('Number of faces detected:', len(faces))

        # Make a copy of the orginal image to draw face detections on
        denoised_image_with_detections = np.copy(denoised_image)

        # Get the bounding box for each detected face
        for (x,y,w,h) in faces:
            # Add a red bounding box to the detections image
            cv2.rectangle(denoised_image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)
            
        return denoised_image_with_detections
    


class blurImage():
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

    def blur(self, image, kernel_width):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 21, 7)
        # Convert the RGB  image to grayscale
        gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 4, 6)
        image_copy = np.copy(image)
        
        kernel = np.ones((kernel_width, kernel_width),np.float32) / (kernel_width*kernel_width)
        blurred_image = cv2.filter2D(image, -1, kernel)

        for (x,y,w,h) in faces:
            padding = 30
            x_start = max(x - padding, 0)
            y_start = max(y - padding, 0)
            x_end = min(x + w + padding, image.shape[1])
            y_end = min(y + h + padding, image.shape[0])
            image_copy[y_start:y_end, x_start:x_end] = cv2.filter2D(blurred_image[y_start:y_end, x_start:x_end], -1, kernel)
            
        return image_copy
    

optimizers = ['SGD','Adam','RMSprop','Adagrad','Adadelta','Adamax','Nadam']
error = {}

class trainModel():
    def __init__(self, epochs):
        self.epochs = epochs
        self.error = error
        self.optimizers = optimizers
        
    def train(self):
        for b in self.optimizers:
            model.compile(loss='mean_squared_error', optimizer = b, metrics = ['mse'])
            regressor = model.fit(X_train, y_train, validation_split=0.2, epochs=self.epochs)
            self.error[b] = regressor.history['val_mean_squared_error']
            
        return self.error
    
    def optimizerLoss(self, error):
        # compare optimizer losses
        loss = []
        plt.figure(figsize=(8,4))
        for key, item in error.items():
            plt.plot(item)
            loss.append(key)
        plt.title('optimizer mean squared error')
        plt.grid()
        plt.legend()
        plt.ylabel('mean squared error')
        plt.xlabel('epoch')
        plt.legend(loss, loc='upper right')
        plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        plt.show()
        
    def buildModel(self, filter_size, dropout, num_dense, dense_layer):
        model = Sequential()
        model.add(Convolution2D(filters=filter_size, kernel_size=(3,3), activation='relu', input_shape=(96,96,1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))
        
        model.add(Convolution2D(filters=filter_size*2, kernel_size=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))
        
        model.add(Convolution2D(filters=filter_size*4, kernel_size=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))
        
        model.add(Flatten())
        
        if num_dense == 1:
            model.add(Dense(dense_layer))
            model.add(Dropout(dropout))

        if num_dense == 2:
            model.add(Dense(dense_layer))
            model.add(Dropout(dropout))
            model.add(Dense(dense_layer))
        else:
            pass
        model.add(Dense(30))
        return model
    
    def buildModelLayers(self, filter_size, dropout, dense_layer):
        model = Sequential()

        model.add(Convolution2D(filters=filter_size, kernel_size=(3,3), activation='relu', input_shape=(96,96,1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(filters=filter_size*2, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(filters=filter_size*4, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(filters=filter_size*8, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(dense_layer, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(30))
        return model
        
        
    def dropModel(self, filter_size, dense_layer):
        model = Sequential()

        model.add(Convolution2D(filters=filter_size, kernel_size=(3,3), activation='relu', input_shape=(96,96,1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Convolution2D(filters=filter_size*2, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Convolution2D(filters=filter_size*4, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(dense_layer, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(30))
        return model    
        
    def visualizeLoss(self, model):
        # loss
        plt.plot(model.history['loss'],linewidth=3,label="train loss")
        plt.plot(model.history['val_loss'],linewidth=3,label="validation loss")
        plt.title('model loss')
        plt.grid()
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        plt.show()

        
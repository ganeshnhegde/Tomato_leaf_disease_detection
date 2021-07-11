# installing Tensarflow GPU

!pip install tensorflow-gpu

# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

##re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = '/content/drive/MyDrive/New Plant Diseases Dataset(Augmented)/train'
valid_path = '/content/drive/MyDrive/New Plant Diseases Dataset(Augmented)/valid'
inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in inception.layers:
    layer.trainable = False

folders = glob('/content/drive/MyDrive/New Plant Diseases Dataset(Augmented)/train/*')
#Add Flatten layer to output of the inception v3
x = Flatten()(inception.output)
#Add dense layer
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=inception.input, outputs=prediction)

#summary of the model
model.summary()

#Compile model
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

#data generating
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

#prepare training set
training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/New Plant Diseases Dataset(Augmented)/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
#prepare validation set
test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/New Plant Diseases Dataset(Augmented)/valid',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=100,
  validation_steps=100
)


# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')#optional

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')#optinal

# save it as a h5 file
from tensorflow.keras.models import load_model
model.save('model_inception.h5')

#prepare data for checking output of the model
finaltest= train_datagen.flow_from_directory('/content/drive/MyDrive/Tomato/dataset/image/FinalModel',
                                                 target_size = (224, 224),
                                                 batch_size = 1,
                                                 shuffle=False,
                                                 class_mode = 'categorical'
                                              )

#classes of the objects
classes = ['Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

#predict using model
a=np.argmax(model.predict(finaltest), axis=1)
print(a)

#Final output
import matplotlib.pyplot as plt
dir1 = '/content/drive/MyDrive/Tomato/dataset/image/FinalModel/1'
dir2 = '/content/drive/MyDrive/Tomato/dataset/image/FinalModel'
import os

for i in range(len(os.listdir(dir1))):
  im = dir2+"/"+finaltest.filenames[i]
  img = image.load_img(im,target_size=(224,224,3))
  plt.imshow(img)
  plt.show()
  print(classes[a[i]])



 

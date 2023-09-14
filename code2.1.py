import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


os.chdir("R:/Face/model")
train= ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)

train_dataset=train.flow_from_directory("R:/Face/model/train/",
                                        target_size=(200,200),
                                        batch_size=60,
                                        class_mode='categorical')

validation_dataset=validation.flow_from_directory("R:/Face/model/validation/",
                                        target_size=(200,200),
                                        batch_size=42,
                                        class_mode='categorical')




model= tf.keras.models.Sequential(
    [
     tf.keras.layers.Conv2D(16,(3,3),activation='relu', input_shape=(200,200,3)),
     tf.keras.layers.MaxPool2D(2,2),
     
     tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
     tf.keras.layers.MaxPool2D(2,2),
     
     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
     tf.keras.layers.MaxPool2D(2,2),    
     
     tf.keras.layers.Flatten(),
     
     tf.keras.layers.Dense(512,activation='relu'),
     
     tf.keras.layers.Dense(3,activation='softmax')
     ]
    
    
    )

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=0.001),metrics=['accuracy'])

model_fit=model.fit(train_dataset,
                    steps_per_epoch=100,
                    epochs=10,
                    validation_data=validation_dataset,
                    validation_steps=100)


model.save('FR.h5')

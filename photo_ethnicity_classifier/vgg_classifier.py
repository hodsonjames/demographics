from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import optimizers

img_folder = 'total_samples'

train_datagen = ImageDataGenerator(
    width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True,
    rotation_range=20, brightness_range=[0.5, 1.5], zoom_range=0.2,
    shear_range=0.2, validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    img_folder, color_mode='rgb', target_size=(150, 150), batch_size=16, subset='training',
    shuffle=True
)

test_generator = train_datagen.flow_from_directory(
    img_folder, color_mode='rgb', target_size=(150, 150), batch_size=16, subset='validation',
    shuffle=True
)

# load model without classifier layers
model = VGG16(include_top=False, input_shape=(150, 150, 3))

# for layer in model.layers:
# 	layer.trainable = False

# add new classifier layers
flat1 = Flatten()(model.output)
class1 = Dense(1024, activation='relu')(flat1)
class1 = Dense(512, activation='relu')(flat1)
output = Dense(4, activation='softmax')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)

# custom optimizer
sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)

#compile model using accuracy to measure model performance
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

epoch_steps = 1024

#train the model
model.fit_generator(train_generator, steps_per_epoch=epoch_steps,
                    validation_data=test_generator, validation_steps=epoch_steps,
                    epochs=1)

model.save('vgg.h5')

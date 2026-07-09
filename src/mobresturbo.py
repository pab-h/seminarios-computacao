from keras.layers import Input
from keras.layers import Dense
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import ReLU
from keras.layers import GlobalAveragePooling2D

from keras.applications import MobileNetV3Small

from keras.applications.mobilenet_v3 import preprocess_input

from keras.models import Model

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    
    if x.shape[-1] != shortcut.shape[-1]:  # Adjust the shortcut dimensions
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

def path1(x):
    x = residual_block(x, 8)
    x = residual_block(x, 16)
    x = residual_block(x, 32)
    x = GlobalAveragePooling2D()(x)
    return x

def create_model(input_shape=(128, 128, 3), num_classes=38):
    inputs = Input(shape=input_shape)
    
    # Path1
    x1 = path1(inputs)
    
    # Path2
    mobilenet = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=input_shape)
    mobilenet.trainable = False

    x2 = preprocess_input(inputs)
    x2 = mobilenet(inputs)
    x2 = GlobalAveragePooling2D()(x2)
    
    # Concatenate paths
    concatenated = Concatenate()([x1, x2])
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(concatenated)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()

    return model
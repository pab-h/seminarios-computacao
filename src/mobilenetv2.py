from keras.layers import Input
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D

from keras.applications              import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input

from keras.models import Model

def create_model(input_shape=(128, 128, 3), num_classes=38):
    inputs = Input(shape=input_shape)

    x = preprocess_input(inputs)

    mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    mobilenet.trainable = False
    
    x = mobilenet(x) 
    x = GlobalAveragePooling2D()(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model
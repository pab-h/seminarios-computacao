# Adaptado de https://github.com/Chiranjit369/Mob-Res/blob/main/Mob-Res_Code.ipynb

import tensorflow as tf
from keras.layers import Layer, Input, Dense, Concatenate, Add, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D
from keras.applications import MobileNetV2
from keras.models import Model

class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size=3, stride=1, **kwargs):

        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        
        # Caminho principal
        self.conv1 = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        
        self.conv2 = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')
        self.bn2 = BatchNormalization()
        
        # Elementos para o ajuste do shortcut (serão construídos dinamicamente no build)
        self.shortcut_conv = None
        self.shortcut_bn = None
        self.add = Add()
        self.relu2 = ReLU()

    def build(self, input_shape):

        # Verifica se o número de canais mudou para ajustar o shortcut
        if input_shape[-1] != self.filters:
            self.shortcut_conv = Conv2D(self.filters, (1, 1), padding='same')
            self.shortcut_bn = BatchNormalization()
        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs):
        shortcut = inputs
        
        # Fluxo principal
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Ajuste do shortcut se necessário
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut)
            
        x = self.add([x, shortcut])
        x = self.relu2(x)
        return x

class Path1(Layer):
    def __init__(self, **kwargs):
        super(Path1, self).__init__(**kwargs)
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(256)
        self.gap = GlobalAveragePooling2D()

    def call(self, inputs):
        x = self.res1(inputs)
        x = self.res2(x)
        x = self.res3(x)
        x = self.gap(x)
        return x

class MobRes(Model):
    def __init__(self, input_shape=(128, 128, 3), num_classes=38, **kwargs):
        super(MobRes, self).__init__(**kwargs)
        
        # Caminho 1
        self.path1 = Path1()
        
        # Caminho 2 (MobileNetV2)
        self.mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        self.gap2 = GlobalAveragePooling2D()
        
        # Junção e Saída
        self.concat = Concatenate()
        self.outputs = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # Executa Path 1
        x1 = self.path1(inputs)
        
        # Executa Path 2
        x2 = self.mobilenet(inputs)
        x2 = self.gap2(x2)
        
        # Concatena e envia para a saída
        concatenated = self.concat([x1, x2])
        return self.outputs(concatenated)

if __name__ == "__main__":
    
    model = MobRes(input_shape=(128, 128, 3), num_classes=38)
    
    import numpy as np

    sample_input = np.random  \
        .rand(1, 128, 128, 3) \
        .astype(np.float32)
    
    _ = model(sample_input)
    
    model.summary()
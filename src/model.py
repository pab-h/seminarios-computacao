import keras

from keras              import layers
from keras.applications import MobileNetV2

class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, stride=1, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
        )
        self.bn2 = layers.BatchNormalization()

        self.shortcut_conv = None
        self.shortcut_bn = None

        self.add = layers.Add()
        self.relu2 = layers.ReLU()

        self.filters = filters

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.shortcut_conv = layers.Conv2D(
                self.filters,
                kernel_size=1,
                padding="same",
            )
            self.shortcut_bn = layers.BatchNormalization()

        super().build(input_shape)

    def call(self, inputs, training=None):
        shortcut = inputs

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut, training=training)

        x = self.add([x, shortcut])
        x = self.relu2(x)

        return x

class CustomPath(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(128)
        self.block3 = ResidualBlock(256)

        self.pool = layers.GlobalAveragePooling2D()

    def call(self, inputs, training=None):

        x = self.block1(inputs, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.pool(x)

        return x

class MobRes(keras.Model):
    def __init__(self, num_classes=38, **kwargs):
        super().__init__(**kwargs)

        self.path1 = CustomPath()

        self.path2 = MobileNetV2(
            weights     = "imagenet",
            include_top = False,
        )

        self.pool = layers.GlobalAveragePooling2D()
        self.concat = layers.Concatenate()
        self.classifier = layers.Dense(
            num_classes,
            activation="softmax",
        )

    def call(self, inputs, training=None):
        x1 = self.path1(inputs, training=training)

        x2 = self.path2(inputs, training=training)
        x2 = self.pool(x2)

        x = self.concat([x1, x2])

        return self.classifier(x)

if __name__ == "__main__":

    model = MobRes(num_classes = 38)

    model.build((None, 128, 128, 3))
    model.summary()

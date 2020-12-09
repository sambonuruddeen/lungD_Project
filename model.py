
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def build_model():
    size = 128
    num_filters = [64, 128, 256, 512]
    inputs = Input((size, size, 1))
    # print(f"the shape of the input images is, {inputs.shape}")

    skip_x = []
    x = inputs
    ## Encoder
    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)

    ## Bridge
    x = conv_block(x, num_filters[-1])

    num_filters.reverse()
    skip_x.reverse()
    ## Decoder
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        x = conv_block(x, f)

    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    # print(f"the shape of x at the end of the network, {x}")
    return Model(inputs, x)

# # Build U-Net model
# inputs = Input((128, 128, 1))
# # s = Lambda(lambda x: x / 255) (inputs)

# c1 = Conv2D(64, (3, 3), activation='relu', padding='same') (inputs)
# c1 = Conv2D(64, (3, 3), activation='relu', padding='same') (c1)
# p1 = MaxPooling2D((2, 2)) (c1)

# c2 = Conv2D(128, (3, 3), activation='relu', padding='same') (p1)
# c2 = Conv2D(128, (3, 3), activation='relu', padding='same') (c2)
# p2 = MaxPooling2D((2, 2)) (c2)

# c3 = Conv2D(256, (3, 3), activation='relu', padding='same') (p2)
# c3 = Conv2D(256, (3, 3), activation='relu', padding='same') (c3)
# p3 = MaxPooling2D((2, 2)) (c3)

# c4 = Conv2D(512, (3, 3), activation='relu', padding='same') (p3)
# c4 = Conv2D(512, (3, 3), activation='relu', padding='same') (c4)
# p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

# c5 = Conv2D(1024, (3, 3), activation='relu', padding='same') (p4)
# c5 = Conv2D(1024, (3, 3), activation='relu', padding='same') (c5)

# u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
# u6 = concatenate([u6, c4])
# c6 = Conv2D(512, (3, 3), activation='relu', padding='same') (u6)
# c6 = Conv2D(512, (3, 3), activation='relu', padding='same') (c6)

# u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
# u7 = concatenate([u7, c3])
# c7 = Conv2D(256, (3, 3), activation='relu', padding='same') (u7)
# c7 = Conv2D(256, (3, 3), activation='relu', padding='same') (c7)

# u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
# u8 = concatenate([u8, c2])
# c8 = Conv2D(128, (3, 3), activation='relu', padding='same') (u8)
# c8 = Conv2D(128, (3, 3), activation='relu', padding='same') (c8)

# u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
# u9 = concatenate([u9, c1], axis=3)
# c9 = Conv2D(64, (3, 3), activation='relu', padding='same') (u9)
# c9 = Conv2D(64, (3, 3), activation='relu', padding='same') (c9)

# outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

# print(f'the shape of the output is {outputs.shape}')

# model = Model(inputs=[inputs], outputs=[outputs])
# # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
# # model.summary()

if __name__ == "__main__":
    model = build_model()
    model.summary()

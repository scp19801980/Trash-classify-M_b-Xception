# M_b Xception, The number of convolution channels of core structure:896
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Activation, Input, Dense, SeparableConv2D
from keras.models import Model
from keras.regularizers import l2


def M_b_Xception_896(input_shape, num_classes):

    l2_reg = 1e-4
    img_input = Input(shape=input_shape)
 
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False,
               kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), use_bias=False,
               kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
 
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
 
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
 
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
 
    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
 
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
 
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
 
    residual = Conv2D(896, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
 
    x = Activation('relu')(x)
    x = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
 
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
 

    x1 = Activation('relu')(x)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)

    residual1 = x1
    x1 = Activation('relu')(x)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = layers.add([x1, residual1])

    residual1 = x1
    x1 = Activation('relu')(x)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = layers.add([x1, residual1])

    residual1 = x1
    x1 = Activation('relu')(x)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = layers.add([x1, residual1])

    residual1 = x1
    x1 = Activation('relu')(x)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = layers.add([x1, residual1])

    residual1 = x1
    x1 = Activation('relu')(x)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = layers.add([x1, residual1])

    residual1 = x1
    x1 = Activation('relu')(x)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = layers.add([x1, residual1])

    residual1 = x1
    x1 = Activation('relu')(x)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x = layers.add([x1, residual1])

    
 
    residual = Conv2D(1024, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
 
    x = Activation('relu')(x)
    x = SeparableConv2D(896, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
 
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
 
    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
 
    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False,
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    
 
    model = Model(img_input, x)
    return model

#input_shape = (229, 229, 3)
#num_classes = 6
#M_b_Xception_896(input_shape, num_classes).summary()
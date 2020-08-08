# M_b Xception, The number of convolution channels of core structure:728
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Activation, Input, Dense, SeparableConv2D
from keras.models import Model
from keras.regularizers import l2


def M_b_Xception_728(input_shape, num_classes):

    l2_reg = 1e-4
    img_input = Input(shape=input_shape)
 
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='Conv2D_32',
               kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='Conv2D_64',
               kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
 
    residual = Conv2D(128, (1, 1), strides=(2, 2), name='Conv2D_oneone_128',
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization(name='BN_oneone_128')(residual)
 
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='SConv2D_128_1',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization(name='BN_128_1')(x)
    x = Activation('relu', name='Act_128_1')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='SConv2D_128_2',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization(name='BN_128_2')(x)
 
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual], name='add_128')
 
    residual = Conv2D(256, (1, 1), strides=(2, 2), name='Conv2D_oneone_256',
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization(name='BN_oneone_256')(residual)
 
    x = Activation('relu', name='Act_256_1')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='SConv2D_256_1',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization(name='BN_256_1')(x)
    x = Activation('relu', name='Act_256_2')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='SConv2D_256_2',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization(name='BN_256_2')(x)
 
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual], name='add_256')
 
    residual = Conv2D(728, (1, 1), strides=(2, 2), name='Conv2D_oneone_728',
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization(name='BN_oneone_728')(residual)
 
    x = Activation('relu', name='Act_728_1')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_728_1',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization(name='BN_728_1')(x)
    x = Activation('relu', name='Act_728_2')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_728_2',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization(name='BN_728_2')(x)
 
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual], name='add_728')
 

    x1 = Activation('relu', name='Act_hx1_1')(x)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx1_1',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx1_1')(x1)
    x1 = Activation('relu', name='Act_hx1_2')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx1_2',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx1_2')(x1)
    x1 = Activation('relu', name='Act_hx1_3')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx1_3',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx1_3')(x1)

    residual1 = x1
    x1 = Activation('relu', name='Act_hx2_1')(x)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx2_1',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx2_1')(x1)
    x1 = Activation('relu', name='Act_hx2_2')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx2_2',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx2_2')(x1)
    x1 = Activation('relu', name='Act_hx2_3')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx2_3',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx2_3')(x1)
    x1 = layers.add([x1, residual1], name='add_hx2')

    residual1 = x1
    x1 = Activation('relu', name='Act_hx3_1')(x)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx3_1',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx3_1')(x1)
    x1 = Activation('relu', name='Act_hx3_2')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx3_2',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx3_2')(x1)
    x1 = Activation('relu', name='Act_hx3_3')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx3_3',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx3_3')(x1)
    x1 = layers.add([x1, residual1], name='add_hx3')

    residual1 = x1
    x1 = Activation('relu', name='Act_hx4_1')(x)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx4_1',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx4_1')(x1)
    x1 = Activation('relu', name='Act_hx4_2')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx4_2',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx4_2')(x1)
    x1 = Activation('relu', name='Act_hx4_3')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx4_3',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx4_3')(x1)
    x1 = layers.add([x1, residual1], name='add_hx4')

    residual1 = x1
    x1 = Activation('relu', name='Act_hx5_1')(x)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx5_1',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx5_1')(x1)
    x1 = Activation('relu', name='Act_hx5_2')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx5_2',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx5_2')(x1)
    x1 = Activation('relu', name='Act_hx5_3')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx5_3',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx5_3')(x1)
    x1 = layers.add([x1, residual1], name='add_hx5')

    residual1 = x1
    x1 = Activation('relu', name='Act_hx6_1')(x)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx6_1',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx6_1')(x1)
    x1 = Activation('relu', name='Act_hx6_2')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx6_2',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx6_2')(x1)
    x1 = Activation('relu', name='Act_hx6_3')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx6_3',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx6_3')(x1)
    x1 = layers.add([x1, residual1], name='add_hx6')

    residual1 = x1
    x1 = Activation('relu', name='Act_hx7_1')(x)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx7_1',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx7_1')(x1)
    x1 = Activation('relu', name='Act_hx7_2')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx7_2',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx7_2')(x1)
    x1 = Activation('relu', name='Act_hx7_3')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx7_3',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx7_3')(x1)
    x1 = layers.add([x1, residual1], name='add_hx7')

    residual1 = x1
    x1 = Activation('relu', name='Act_hx8_1')(x)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx8_1',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx8_1')(x1)
    x1 = Activation('relu', name='Act_hx8_2')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx8_2',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx8_2')(x1)
    x1 = Activation('relu', name='Act_hx8_3')(x1)
    x1 = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_hx8_3',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(name='BN_hx8_3')(x1)
    x = layers.add([x1, residual1], name='add_hx8')

    
 
    residual = Conv2D(1024, (1, 1), strides=(2, 2), name='Conv2D_oneone_1024',
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization(name='BN_oneone_1024')(residual)
 
    x = Activation('relu', name='Act_1024_1')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='SConv2D_1024_1',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization(name='BN_1024_1')(x)
    x = Activation('relu', name='Act_1024_2')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='SConv2D_1024_2',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization(name='BN_1024_2')(x)
 
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual], name='add_1024')
 
    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='SConv2D_2048_1',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization(name='BN_2048_1')(x)
    x = Activation('relu', name='Act_2048_1')(x)
 
    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='SConv2D_2048_2',
                        kernel_regularizer=l2(l2_reg), kernel_initializer='he_normal')(x)
    x = BatchNormalization(name='BN_2048_2')(x)
    x = Activation('relu', name='Act_2048_2')(x)
 
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    
 
    model = Model(img_input, x)
    return model

#input_shape = (229, 229, 3)
#num_classes = 6
#M_b_Xception_728(input_shape, num_classes).summary()
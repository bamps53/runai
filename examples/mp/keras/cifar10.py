from __future__ import print_function

import runai.mp

runai.mp.init(splits=2, method=runai.mp.Method.Cout)
#runai.mp.init(splits=2, method=runai.mp.Method.Cin)


import keras
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
import click
import time


num_classes = 10


# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------




def concat_ones(src):
    l = list(src.shape)
    l[3] = 1
    shape = tuple(l)
    ones = np.zeros(shape)
    return np.concatenate((src, ones), 3)



def get_data(subtract_pixel_mean):

    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()


    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean



    x_train = concat_ones(x_train)
    x_test = concat_ones(x_test)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train,y_train,x_test,y_test



'''

def lr_schedule(epoch, lr):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    if epoch == 181:
        lr *= 0.5
    elif epoch == 161:
        lr *= 1e-1
    elif epoch == 121:
        lr *= 1e-1
    elif epoch == 81:
        lr *= 1e-1

    print('Learning rate: ', lr)
    return lr
'''

def lr_schedule_with_init(epoch,init_lr, lr):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr1 = init_lr
    if epoch > 180:
        lr1 *= 0.5e-3
    elif epoch > 160:
        lr1 *= 1e-3
    elif epoch > 120:
        lr1 *= 1e-2
    elif epoch > 80:
        lr1 *= 1e-1
    if True:# lr > lr1:
        lr = lr1
    print('Learning rate: ', lr)
    return lr


class StepTimeReporter(keras.callbacks.Callback):
    def __init__(self):
        self.start = 0

    def on_batch_begin(self, batch, logs={}):
        self.start = time.time()

    def on_batch_end(self, batch, logs={}):
        print(' >> Step %d took %g sec' % (batch, time.time() - self.start))

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet50_builtin():
    model = ResNet50(include_top=True, weights=None, classes=10,input_shape=[32,32,3])
    return model


def get_model(version, input_shape, depth, lr):

    if version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth)
    if version == 1:
        model = resnet_v1(input_shape=input_shape, depth=depth)
    if version == 3:
        model = resnet50_builtin()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])
    model.summary()
    return model


def run(epochs=200, batch_size=32, init_lr=1e-3, version=2, n=6, subtract_pixel_mean=True, data_augmentation=True):



    # Get Data
    # Subtracting pixel mean improves accuracy
    x_train,y_train,x_test,y_test = get_data(subtract_pixel_mean)
    # Input image dimensions.
    input_shape = x_train.shape[1:]



    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
        model_type = 'ResNet%dv%d' % (depth, version)
    elif version == 2:
        depth = n * 9 + 2
        model_type = 'ResNet%dv%d' % (depth, version)
    elif version == 3:
        model_type='ResNet50builtin'
        depth = 0
    print(model_type)

    # Get Model
    model = get_model(version, input_shape, depth, init_lr)

    # Tensorboard location
    if  os.getenv('RUNAI_TENSORBOARD_DIR'):
        outputs_dir = '/runai/outputs/'
    else:
        outputs_dir = '/tmp/'


    # Prepare model model saving directory.
    save_dir = os.path.join(outputs_dir, 'saved_models')
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_schedule = lambda epoch,lr: lr_schedule_with_init(epoch,init_lr,lr)
    lr_scheduler = LearningRateScheduler(lr_schedule,verbose=1)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6,
                                   verbose=1)

    cur_output_dir = os.getenv('RUNAI_TENSORBOARD_DIR')#RUNAI_TENSORBOARD_DIR #‘./output’
    if not cur_output_dir:
        cur_output_dir='/tmp/'
    tensor_board = TensorBoard(log_dir=os.path.join(cur_output_dir, 'tensor_board'),
                               histogram_freq=0,
                               write_graph=False,
                               write_images=False)

    time_reporter = StepTimeReporter()
    callbacks = [checkpoint, lr_reducer, lr_scheduler, time_reporter, tensor_board]

    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        preprocess_function = None
        if version == 3:
            preprocess_function = preprocess_input
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=preprocess_function,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        steps_per_epoch = np.ceil(len(y_train) / float(batch_size))
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


    # exercise different output information and formats

    save_dir = os.path.join(outputs_dir, 'saved_weights')
    model_name = 'cifar10_%s_model.weights.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    model.save_weights(filepath)


    with open(os.path.join(outputs_dir,"model.json"), "w") as f:
        f.write(model.to_json())

    with open(os.path.join(outputs_dir,"model.yaml"), "w") as f:
        f.write(model.to_yaml())




@click.command()
@click.option('--epochs',
    default=200,
    show_default=True,
    help='epochs',
    type=int
)
@click.option(
    '--batch_size',
    default=32,
    show_default=True,
    help='batch_size',
    type=int
)
@click.option('--lr',
    default=1e-3,
              show_default=True,
    help='lr',
    type=float)
@click.option(
    '--version',
    default='2',
    type=click.Choice(['1','2','3']),
)
@click.option('--n',
    type=click.Choice(['2','3','5','6','7','9','12','18','27','111']),
    default='2',
    show_default=True
)
@click.option(
    '--subtract_pixel_mean/--no-subtract_pixel_mean',
    default         = True,
    is_flag         = True,
    show_default    = True,
    help            = 'subtract_pixel_mean',
)
@click.option(
    '--data_augmentation/--no-data-augmentation',
    default         = True,
    is_flag         = True,
    show_default    = True,
    help            = '',
)



def main(epochs, batch_size, lr, version, n, subtract_pixel_mean, data_augmentation):
    run(epochs=epochs,
        batch_size=batch_size,
        init_lr=lr,
        version=int(version),
        n=int(n),
        subtract_pixel_mean=subtract_pixel_mean,
        data_augmentation=data_augmentation)


if __name__ == "__main__":
    main()

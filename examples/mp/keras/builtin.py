import sys

import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize

import runai.mp

runai.mp.init(splits=2, method=runai.mp.Method.Cout)

if len(sys.argv) == 1:
    MODEL = 'vgg16'
else:
    MODEL = sys.argv[1]

def resize_images(src, shape, mode):
    resized = [imresize(img, shape, 'bilinear', mode) for img in src]
    return np.stack(resized)

def concat_ones(src):
    l = list(src.shape)
    l[3] = 1
    shape = tuple(l)
    ones = np.zeros(shape)
    return np.concatenate((src, ones), 3)
    
def cifar10_data(train_samples, test_samples, num_classes, data_format='channels_last', num_channels=3, trg_image_dim_size=0):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    print('Loaded train samples')
    
    if train_samples > 0:
        x_train = x_train[:train_samples]
        y_train = y_train[:train_samples]
    
    if test_samples > 0:
        x_test = x_test[:test_samples]
        y_test = y_test[:test_samples]
            
    if num_channels == 4:
        x_train = concat_ones(x_train)
        x_test = concat_ones(x_test)
    
    if trg_image_dim_size > 0:
        mode = 'RGBA' if num_channels == 4 else 'RGB'
        x_train = resize_images(x_train, (trg_image_dim_size, trg_image_dim_size), mode)
        x_test = resize_images(x_test, (trg_image_dim_size, trg_image_dim_size), mode)
    
    if data_format == 'channels_first':
        x_train = np.transpose(x_train, (0, 3, 1, 2))
        x_test = np.transpose(x_test, (0, 3, 1, 2))
        keras.backend.set_image_data_format(data_format)
        
    y_train = np.clip(y_train, None, num_classes - 1)
    y_test = np.clip(y_test, None, num_classes - 1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print('Preprocessed train samples')
    print('  X train shape: %s' % str(x_train.shape))
    print('  Y train shape: %s' % str(y_train.shape))
    print('  X test shape: %s' % str(x_test.shape))
    print('  Y test shape: %s' % str(y_test.shape))
    
    return (x_train, y_train), (x_test, y_test)
    
models = {
    'xception':             'Xception',
    'vgg16':                'VGG16',
    'vgg19':                'VGG19',
    'inception_v3':         'InceptionV3',
    'inception_resnet_v2':  'InceptionResNetV2',
    'mobilenet':            'MobileNet',
    'densenet':             'DenseNet169',
    'nasnet':               'NASNetLarge',
    'mobilenet_v2':         'MobileNetV2',
    'resnet50':             'ResNet50',
}

(x_train, y_train), (x_test, y_test) = cifar10_data(
    train_samples=100,
    test_samples=100,
    num_classes=10,
    trg_image_dim_size=300,
    data_format='channels_last',
    num_channels=3 if runai.mp.method == runai.mp.Method.Cout else 4
)

module = getattr(keras.applications, MODEL)
func = getattr(module, models[MODEL])

model = func(
    input_shape=x_train[0].shape,
    include_top=True,
    weights=None,
    input_tensor=None,
    pooling=None,
    classes=10)
        
print('%s model created' % MODEL)
    
model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.SGD(lr=1e-3),
                metrics=['accuracy'])

train_datagen = ImageDataGenerator()
train_datagen.fit(x_train)

val_datagen = ImageDataGenerator()
val_datagen.fit(x_test)

BATCH_SIZE = 2

model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=len(x_train)/BATCH_SIZE,
            epochs=2,
            validation_steps=len(x_test)/BATCH_SIZE,
            validation_data=val_datagen.flow(x_test, y_test),
            shuffle=False)

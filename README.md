# Keras
ResNet50 defect detect


代码结构：
DATASET_PATH  = './train'
IMAGE_SIZE    = (256, 256)
NUM_CLASSES   = 2
BATCH_SIZE    = 8  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS    = 10
WEIGHTS_FINAL = 'model-resnet50-final.h5'

根目录： './train'
              目录结构：



图像大小：256X256
深度学习由于数据集增强（没有足够的训练集，通常都会这么做），考虑内存占用以及没有GPU，所以会将一张大图剪切成小图进行二分类训练：
    对检测区域进行剪切（halcon）：

read_image (Ng1, 'E:/Keras-test/image/NG/256/ng (67).jpg')
*选出检测区域
threshold(Ng1, Region, 128, 255)
connection(Region, ConnectedRegions)
select_shape(ConnectedRegions, SelectedRegions, 'area', 'and', 15000, 999999999)
fill_up(SelectedRegions, RegionFillUp)
reduce_domain(Ng1, RegionFillUp, ImageReduced1)
crop_domain(ImageReduced1, ImagePart1)
get_image_size(ImagePart1, Width, Height)


indexW:=Width/256
indexH:=Height/256

for j := 1 to indexW by 1
    for i := 1 to indexH by 1
        gen_rectangle1(Rectangle, (i-1)*255, (j-1)*255, i*255, j*255)
        reduce_domain(ImagePart1, Rectangle, ImageReduced)
        crop_domain(ImageReduced, ImagePart)
        str:='test_image/'+i+'_'+j+'.jpg'
        write_image(ImagePart, 'jpeg', 0, str)
    endfor
endfor

数据集增强（旋转90、180、270、镜像）：


list_files ('E:/Keras-test/image/NG/256NG', ['files','follow_links'], ImageFiles)
tuple_regexp_select (ImageFiles, ['\\.(jpg|jpeg|jp2)$','ignore_case'], ImageFiles)
for Index := 0 to |ImageFiles| - 1 by 1
    read_image (Image, ImageFiles[Index])
    rotate_image(Image, ImageRotate90, 90, 'constant')
    mirror_image(ImageRotate90, ImageMirror90r, 'row')
    mirror_image(ImageRotate90, ImageMirror90c, 'column')
    str:=ImageFiles[Index]+'90.jpg'
        write_image(ImageRotate90, 'jpeg', 0, str)
    str:=ImageFiles[Index]+'90r.jpg'
        write_image(ImageMirror90r, 'jpeg', 0, str)
    str:=ImageFiles[Index]+'90c.jpg'
        write_image(ImageMirror90c, 'jpeg', 0, str) 
    
        
        
    rotate_image(Image, ImageRotate180, 180, 'constant')
    mirror_image(ImageRotate180, ImageMirror180r, 'row')
    mirror_image(ImageRotate180, ImageMirror180c, 'column')
    str:=ImageFiles[Index]+'180.jpg'
        write_image(ImageRotate180, 'jpeg', 0, str)
    str:=ImageFiles[Index]+'180r.jpg'
        write_image(ImageMirror180r, 'jpeg', 0, str)
    str:=ImageFiles[Index]+'180c.jpg'
        write_image(ImageMirror180c, 'jpeg', 0, str)
    
    
    
    rotate_image(Image, ImageRotate270, 270, 'constant')
    mirror_image(ImageRotate270, ImageMirror270r, 'row')
    mirror_image(ImageRotate270, ImageMirror270c, 'column')
    str:=ImageFiles[Index]+'270.jpg'
        write_image(ImageRotate270, 'jpeg', 0, str)
    str:=ImageFiles[Index]+'270r.jpg'
        write_image(ImageMirror270r, 'jpeg', 0, str)
    str:=ImageFiles[Index]+'270c.jpg'
        write_image(ImageMirror270c, 'jpeg', 0, str)
    
    
endfor

NUM_CLASSES   = 2    #二分类
NUM_EPOCHS    = 10  #10个循环训练
WEIGHTS_FINAL = 'model-resnet50-final.h5'  #第一次会网上下载网络结构，训练完成后会保存更新权值的网络结构。

训练：
'''
'''


from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from polit import LossHistory

DATASET_PATH  = './train'
IMAGE_SIZE    = (256, 256)
NUM_CLASSES   = 2
BATCH_SIZE    = 8  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS    = 10
WEIGHTS_FINAL = 'model-resnet50-final.h5'

#创建一个实例LossHistory
history = LossHistory()

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/validation',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

# show class indices
print('****************')
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')

# build our classifier model based on pre-trained ResNet50:
# 1. we don't include the top (fully connected) layers of ResNet50
# 2. we add a DropOut layer followed by a Dense (fully connected)
#    layer which generates softmax class score for each class
# 3. we compile the final model using an Adam optimizer, with a
#    low learning rate (since we are 'fine-tuning')
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True
net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
print(net_final.summary())

# train the model
net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches, 
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS,
                        callbacks=[history])

# save trained weights
net_final.save(WEIGHTS_FINAL)
#绘制acc-loss曲线
history.loss_plot('epoch')

测试：
'''
This script goes along my blog post:
"Keras Cats Dogs Tutorial" (https://jkjung-avt.github.io/keras-tutorial/)
'''


from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing import image
import os
import sys
import glob
import argparse
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    return args


def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'))
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    files = [f for f in files if f.endswith('JPG') or f.endswith('jpg')]

    if not len(files):
        sys.exit('No images found by the given path!')

    return files


if __name__ == '__main__':
    # args = parse_args()
    files = get_files('./test')
    cls_list = ['NG', 'OK']

    # load the trained model
    net = load_model('model-resnet50-final.h5')#./test/model-resnet50-final.h5

    # loop through all files and make predictions
    for f in files:
        img = image.load_img(f, target_size=(256,256))
        if img is None:
            continue
        x = image.img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        pred = net.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        print(f)
        for i in top_inds:

            print('    {:.3f}  {}'.format(pred[i], cls_list[i]))

        plt.imshow(img)
        plt.axis('off')
        # ax.text(tx, ty, index, color='white', ha=halign, va='center',
        #         bbox={'boxstyle': 'square', 'facecolor': facecolor})
        # ec = (1., 0.5, 0.5), fc = (1., 0.8, 0.8), )
        plt.text(5, 20, f + '  :' + str('  {:.3f}  {} / {:.3f}  {}'.format(pred[0], cls_list[0],pred[1], cls_list[1])),
                 ha="left", va="center",bbox=dict(boxstyle="round", ec='red', fc='white', ))
        plt.show()

loos/acc训练结果绘图：
import matplotlib.pyplot as plt
import keras

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')#plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')#给x，y轴加注释
        plt.legend(loc="upper right")#设置图例显示位置
        plt.show()

检测结果：






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
![image](https://github.com/KangWang1991/Keras/blob/master/images/clipboard.png)


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

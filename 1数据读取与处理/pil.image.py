#由于transforms的输入规定为PIL文件格式，所以我们需要使用PIL.Image模块
import PIL.Image as Image
import os
from torchvision import transforms as transforms

#读取图片
outfile = './samples'
im = Image.open('./test.jpg')
im.save(os.path.join(outfile, 'test.jpg'))

# 使用Compose函数生成一个PiPeLine，
# 经过这样处理后，我们就可以直接使用data_transform来进行图像的变换
data_transform={'train':transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])},


#随机比例缩放
new_im = transforms.Resize((100, 200))(im)
print(f'{im.size}---->{new_im.size}')
new_im.save(os.path.join(outfile, '1.jpg'))

# 随机位置裁剪
new_im = transforms.RandomCrop(100)(im)   # 裁剪出100x100的区域
new_im.save(os.path.join(outfile, '2_1.jpg'))
new_im = transforms.CenterCrop(100)(im)
new_im.save(os.path.join(outfile, '2_2.jpg'))

# 随机水平/垂直翻转
new_im = transforms.RandomHorizontalFlip(p=1)(im)   # p表示概率
new_im.save(os.path.join(outfile, '3_1.jpg'))
new_im = transforms.RandomVerticalFlip(p=1)(im)
new_im.save(os.path.join(outfile, '3_2.jpg'))

# 随机角度旋转
new_im = transforms.RandomRotation(45)(im)    #随机旋转45度
new_im.save(os.path.join(outfile, '4.jpg'))

# 色度、亮度、饱和度、对比度的变化
new_im = transforms.ColorJitter(brightness=1)(im)
new_im = transforms.ColorJitter(contrast=1)(im)
new_im = transforms.ColorJitter(saturation=0.5)(im)
new_im = transforms.ColorJitter(hue=0.5)(im)
new_im.save(os.path.join(outfile, '5_1.jpg'))

# 进行随机的灰度化
new_im = transforms.RandomGrayscale(p=0.5)(im)    # 以0.5的概率进行灰度化
new_im.save(os.path.join(outfile, '6_2.jpg'))

# Padding (将原始图padding成正方形)
new_im = transforms.Pad((0, (im.size[0]-im.size[1])//2))(im)  # 原图为（500,313）
new_im.save(os.path.join(outfile, '7.jpg'))

# 使用Compose函数生成一个PiPeLine，
# 经过这样处理后，我们就可以直接使用data_transform来进行图像的变换
data_transform={'train':transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),

# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

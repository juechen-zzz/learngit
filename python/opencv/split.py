import os
import shutil


source_dir ='/home/test/nhp/Inpainting/AIM20'
annotion_dir = '/home/test/nhp/Inpainting/ADK_val'

img = os.listdir(source_dir)
keyword = "with"

count = 0

#遍历文件夹
for fileNum in img:
    if fileNum.find(keyword) != -1:
        imgPath = os.path.join(source_dir,fileNum)
        shutil.copy(imgPath,annotion_dir)
        count += 1
        print(count)

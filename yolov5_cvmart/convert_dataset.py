# #  思路
# 转换标签的同时把图片和标签按分类分开
# 按比例划分数据集
# 
import os
import shutil
from os import listdir, getcwd
from os.path import join
import random
import argparse
import xml.etree.ElementTree as ET
classes= ['person','head','door_open','door_half_open','door_close','daizi','bottle','bag','box','plastic_basket','suitcase','mobile_phone','umbrella','folder','bicycle','electric_scooter']    #自己训练的类别
datasets_path = '/home/data/'  # 新划分数据集地址
from glob import glob
seed = 42
data_root_path = glob('/home/data/**/**/')
if len(data_root_path) == 0:
    data_root_path = glob('/home/data/**/')
# print(data_root_path)
# data_root_path = os.listdir(datasets_path)
# data_root_path = ['/home/data/888/', '/home/data/1001/', '/home/data/901/']  # 原始数据集地址
num_class = [0 for _ in range(len(classes))]
num_class_dict = {}

# 准换标签xml --> txt
def convert_annotation(xml_path,txt_path):
    in_file = open(xml_path,encoding='utf-8')
    out_file = open(txt_path, 'w',encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    class_txt = []
    for obj in root.iter('object'):
        difficult = 0
        cls = obj.find('name').text
        class_txt.append(cls)
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        global num_class
        num_class[cls_id] += 1
        xmlbox = obj.find('bndbox')
        b=(float(xmlbox.find('xmin').text),float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
        float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()
    # print(class_txt,'1')
    return class_txt
# 将jpg和xml分离
def jpg_xml(data_root_path):
    if not os.path.exists(datasets_path + 'data/'):
        os.makedirs(datasets_path + 'data/')
    data_path  = datasets_path + 'data/'
    
    for class_name in classes:
        if not os.path.exists(data_path + class_name):
            os.makedirs(data_path + class_name)
        if not os.path.exists(data_path + class_name + '/labels/'):
            os.makedirs(data_path + class_name + '/labels/')
        if not os.path.exists(data_path + class_name + '/images/'):
            os.makedirs(data_path + class_name + '/images/')
        
        
    xml_path_list = glob(data_root_path+'*.xml')
    
    for xml_path in xml_path_list:
        name = xml_path.split('/')[-1].split('.')[0]
        # print(xml_path)
        txt_path = xml_path.split('.')[0] + '.txt'
        
        class_txt = convert_annotation(xml_path,txt_path)

        jpg_path = xml_path.split('.')[0] + '.jpg'
        # print(data_root_path)
        dir_path = data_root_path.split('/')[-2]
        # print(dir_path)
        for class_name in class_txt:
            dst_jpg_path = data_path + class_name + '/images/' + name+ '_' + dir_path + '.jpg'
            # print(dst_jpg_path)
            dst_txt_path = data_path + class_name + '/labels/' + name + '_' + dir_path +'.txt'
            shutil.copyfile(jpg_path,dst_jpg_path)
            shutil.copyfile(txt_path,dst_txt_path)
        
    # filelist = os.listdir(data_root_path)
    # for files in filelist:
    #     filename1 = os.path.splitext(files)[1]  # 读取文件后缀名
    #     # name = data_root_path.split('/')[3]
    #     if filename1 == '.jpg':
    #         full_path = os.path.join(data_root_path, files)
    #         shutil.move(full_path, datasets_path+'images/'+str(num)+'_'+files)
    #     elif filename1 == '.xml':
    #         full_path = os.path.join(data_root_path, files)
    #         shutil.move(full_path, datasets_path+'Annotations/'+str(num)+'_'+files)
    #     else :
    #         continue


# 划分数据集
def train_val_split(): #运行此函数
    trainval_percent = 1
    train_percent = 9
    txtsavepath = datasets_path
    ftrain = open(txtsavepath + 'train.txt', 'w+')
    ftest = open(txtsavepath +  'test.txt', 'w+')
    fval = open(txtsavepath + 'val.txt', 'w+')
    read_name = []
    global num_class_dict
    for num,class_name in zip(num_class, classes):
        num_class_dict.update({class_name:num})
        # print(num_class_dict)
        
    num_class_dict = dict(sorted(num_class_dict.items(), key=lambda x: x[1]))
    # print(num_class_dict)
    
    for class_name in list(num_class_dict.keys()):
        # print(class_name)
        images_filepath = datasets_path + 'data/' + class_name + '/images/'
        total_imgfiles = os.listdir(images_filepath)
        num = len(total_imgfiles)
        # lists = range(num)
        tr = int(num * train_percent)
        total_imgfiles.sort()
        # random.seed(seed)
        # train = random.sample(lists, tr)
        # for j in lists:
 
        for j in range(num):
            filename = total_imgfiles[j]
            if filename not in read_name:
                read_name.append(filename)
                filepath = images_filepath + filename + '\n'
                if j%10<train_percent:
                    if class_name == "umbrella":
                        for i in range(10):
                            fval.write(filepath)
                            ftrain.write(filepath)
                    ftrain.write(filepath)
                else:
                    fval.write(filepath)
                    ftest.write(filepath)

    
    ftrain.close()
    fval.close()
    ftest.close()


# 转换box
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# 转换标签
# def convert_annotation(image_id):
#     in_file = open(datasets_path + 'Annotations/%s.xml' % (image_id),encoding='utf-8')
#     out_file = open(datasets_path + 'labels/%s.txt' % (image_id), 'w',encoding='utf-8')

#     tree = ET.parse(in_file)
#     root = tree.getroot()
#     size = root.find('size')
#     w = int(size.find('width').text)
#     h = int(size.find('height').text)

#     for obj in root.iter('object'):
#         difficult = 0
#         cls = obj.find('name').text
#         if cls not in classes or int(difficult) == 1:
#             continue
#         cls_id = classes.index(cls)
#         global num_class
#         num_class[cls_id] += 1
#         xmlbox = obj.find('bndbox')
#         b=(float(xmlbox.find('xmin').text),float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
#         float(xmlbox.find('ymax').text))
#         bb = convert((w, h), b)
#         out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# 生成新的标签
# def generate_labels():  # 运行此函数
#     if not os.path.exists(datasets_path + 'labels/'):
#         os.makedirs(datasets_path + 'labels/')
#     sets = ['train', 'val']
#     for image_set in sets:
#         image_ids = open(datasets_path + '%s.txt' % (image_set)).read().strip().split('\n')
#         for image_id in image_ids:
#             if image_id.split('/')[-1][:-4] != '':
#                 convert_annotation(image_id.split('/')[-1][:-4])
                
            

if __name__ == "__main__":

#     判断是否已经移动过数据
    if not os.path.exists(datasets_path + 'data/'):
# 等区间均匀采样
        for dir_path in data_root_path:
            jpg_xml(dir_path)
    train_val_split()
    print("各类别数量：",num_class)
            
    

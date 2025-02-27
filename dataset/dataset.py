from torch import nn
from torch.utils.data import Dataset
import csv
import os
import sys
import re
from PIL import Image
import cv2
from torchvision import transforms
import torch
import json
from transformers import BertTokenizer, BertModel
import random
from nltk.corpus import wordnet
import nltk
import pandas as pd
# nltk.download('wordnet'


class BertEncoder(nn.Module):
    def __init__(self, num_labels=2):  # num_labels 代表分类的数量
        super(BertEncoder, self).__init__()
        # 使用预训练的BERT模型
        self.bert = BertModel.from_pretrained('./model/BERT_pretain')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)  # 用于分类的层

    def forward(self, input_ids, attention_mask=None):
        # 获取BERT的输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 获取[CLS]标记的表示
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)  # 用分类层进行预测
        return logits


def csv_to_dict(csv_file_path):
    """
    将CSV文件读取为字典形式
    :param csv_file_path: CSV文件路径
    :return: 返回一个包含每一行数据的字典列表
    """
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)  # 使用DictReader将每行数据读取为字典
        image_name = []
        labels = []
        for row in reader:
            try:
                image_name.append(row["image_train"])
                labels.append(row["label_train"])
            except:
                image_name.append(row["image_test"])
                labels.append(row["label_test"])
    return image_name, labels


class BaseDatasetTrain(Dataset):
    def __init__(self):
        self.image_path='./data/00-HAM10000/train/images'
        label_path = ['./data/00-HAM10000/label/split_data_1_fold_train.csv', './data/00-HAM10000/label/split_data_1_fold_test.csv']
        self.images_name1, self.labels1 = csv_to_dict(label_path[0])
        self.images_name2, self.labels2 = csv_to_dict(label_path[1])
        self.images_names = self.images_name1 + self.images_name2
        # 这里把其他的数据集也加进来
        
        self.labels = self.labels1+ self.labels2
        self.label_dict = {}
        self.image_paths = [os.path.join(self.image_path, name) for name in os.listdir(self.image_path)]
        data_path = [
            '/mnt/data2/zzixuantang/SAM2-Unet/data/00-data_processed/00-vertebra/00-MR_AVBCE/train/images',
            '/mnt/data2/zzixuantang/SAM2-Unet/data/00-data_processed/00-vertebra/01-verse20/train/images',
            '/mnt/data2/zzixuantang/SAM2-Unet/data/00-data_processed/02-polyp/00-mixeddata/train/images',
            '/mnt/data2/zzixuantang/SAM2-Unet/data/00-data_processed/03-lungCT/00-ieee8023/train/images',
            '/mnt/data2/zzixuantang/SAM2-Unet/data/00-data_processed/04-lungInfectionCT/00-ieee8023/train/images'
        ]
        label_list = [7,7,8,9,10]
        for data_path1, label in zip(data_path, label_list):
            for image_name2 in os.listdir(data_path1):
                print(image_name2)
                self.image_paths.append(os.path.join(data_path1,image_name2))
                self.label_dict[image_name2] = label
        
        for image_name, label in zip(self.images_names, self.labels):
            self.label_dict[image_name] = label

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪到224x224大小
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),              # 转换为Tensor格式
        ])
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1000 
        image_path = self.image_paths[idx]
        # # 查询这个图片的标签
        image_name = os.path.basename(image_path)
        label = self.label_dict[image_name]
        # image duqu 


        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        transformed_image = self.transform(image)

        return transformed_image, int(label)
        
    
    
class BaseDatasetTest(Dataset):
    def __init__(self):
        self.image_path='./data/00-HAM10000/test/images'
        label_path = ['./data/00-HAM10000/label/split_data_1_fold_train.csv', './data/00-HAM10000/label/split_data_1_fold_test.csv']
        self.images_name1, self.labels1 = csv_to_dict(label_path[0])
        self.images_name2, self.labels2 = csv_to_dict(label_path[1])
        self.images_names = self.images_name1 + self.images_name2
        self.labels = self.labels1+ self.labels2
        self.label_dict = {}
        self.image_paths = [os.path.join(self.image_path, name) for name in os.listdir(self.image_path)]
        data_path = [
            '/mnt/data2/zzixuantang/SAM2-Unet/data/00-data_processed/00-vertebra/00-MR_AVBCE/test/images',
            '/mnt/data2/zzixuantang/SAM2-Unet/data/00-data_processed/00-vertebra/01-verse20/test/images',
            '/mnt/data2/zzixuantang/SAM2-Unet/data/00-data_processed/02-polyp/00-mixeddata/test/CVC-ColonDB/images',
            '/mnt/data2/zzixuantang/SAM2-Unet/data/00-data_processed/03-lungCT/00-ieee8023/test/images',
            '/mnt/data2/zzixuantang/SAM2-Unet/data/00-data_processed/04-lungInfectionCT/00-ieee8023/test/images'
        ]
        label_list = [7,7,8,9,10]
        for data_path1, label in zip(data_path, label_list):
            for image_name2 in os.listdir(data_path1):
                print(image_name2)
                self.image_paths.append(os.path.join(data_path1,image_name2))
                self.label_dict[image_name2] = label
        
        for image_name, label in zip(self.images_names, self.labels):
            self.label_dict[image_name] = label
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪到224x224大小
            # transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),              # 转换为Tensor格式
        ])

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # 查询这个图片的标签
        image_name = os.path.basename(image_path)
        label = self.label_dict[image_name]
        # image duqu 
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        transformed_image = self.transform(image)
        return transformed_image, int(label)
        

    
    def __len__(self):
        return len(self.image_paths)
        



class TextImageDatasetTest(Dataset):
    def __init__(self, json_data):
        self.image_path='./data/00-HAM10000/test/images'
        label_path = ['./data/00-HAM10000/label/split_data_1_fold_train.csv', './data/00-HAM10000/label/split_data_1_fold_test.csv']
        self.images_name1, self.labels1 = csv_to_dict(label_path[0])
        self.images_name2, self.labels2 = csv_to_dict(label_path[1])
        self.images_names = self.images_name1 + self.images_name2
        self.labels = self.labels1+ self.labels2
        self.label_dict = {}
        self.image_paths = [os.path.join(self.image_path, name) for name in os.listdir(self.image_path)]

        self.text_path = json_data
        with open(self.text_path , 'r') as jsonf:
            text_dict = json.load(jsonf)
        
        self.tokenizer = BertTokenizer.from_pretrained('./model/BERT_pretain')

        self.texts = []
        for path in self.image_paths:
            self.texts.append(self.remove_chinese_and_punctuation(text_dict[path]))

        for image_name, label in zip(self.images_names, self.labels):
            self.label_dict[image_name] = label
        # # ****************************************************************

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪到224x224大小
            # transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),              # 转换为Tensor格式
        ])
    
    def remove_chinese_and_punctuation(self, text):
        # 正则表达式：匹配所有中文字符和中文标点符号
        # \u4e00-\u9fa5：匹配常用汉字
        # \u3000-\u303F：匹配中文标点符号
        # \uff00-\uffef：匹配全角字符，包括一些标点
        return re.sub(r'[\u4e00-\u9fa5\u3000-\u303F\uff00-\uffef]', '', text)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # 查询这个图片的标签
        image_name = os.path.basename(image_path)
        label = self.label_dict[image_name]
        # image duqu 
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        transformed_image = self.transform(image)


        # 下面处理文本
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )

        return {
            'transformed_image': transformed_image,
            'label': int(label),
            'input_ids': encoding['input_ids'].squeeze(0),  # 去掉batch维度
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        

    
    def __len__(self):
        return len(self.image_paths)
        


class TextImageDatasetTrain(Dataset):
    def __init__(self, json_data):
        self.image_path='./data/00-HAM10000/train/images'
        label_path = ['./data/00-HAM10000/label/split_data_1_fold_train.csv', './data/00-HAM10000/label/split_data_1_fold_test.csv']
        self.images_name1, self.labels1 = csv_to_dict(label_path[0])
        self.images_name2, self.labels2 = csv_to_dict(label_path[1])
        self.images_names = self.images_name1 + self.images_name2
        self.labels = self.labels1+ self.labels2
        self.label_dict = {}
        self.image_paths = [os.path.join(self.image_path, name) for name in os.listdir(self.image_path)]

        self.text_path = json_data
        with open(self.text_path , 'r') as jsonf:
            text_dict = json.load(jsonf)
        
        self.tokenizer = BertTokenizer.from_pretrained('./model/BERT_pretain')

        self.texts = []
        for path in self.image_paths:
            self.texts.append(self.remove_chinese_and_punctuation(text_dict[path]))

        for image_name, label in zip(self.images_names, self.labels):
            self.label_dict[image_name] = label
        # # ****************************************************************

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),          # 随机裁剪到224x224大小
            transforms.RandomHorizontalFlip(),          # 随机水平翻转
            transforms.RandomRotation(30),              # 随机旋转图像，范围为-30到30度
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机调整亮度、对比度、饱和度和色调
            transforms.RandomGrayscale(p=0.1),          # 以10%的概率将图像转换为灰度图像
            transforms.RandomVerticalFlip(p=0.1),       # 以10%的概率进行垂直翻转
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移图像
            transforms.ToTensor(),                      # 转换为Tensor格式
        ])

    def random_deletion(self, text, p=0.2):
        if len(text) <100:
            return text
        """Randomly delete words from the text"""
        words = text.split()
        if len(words) == 1:
            return text  # Prevent deletion if only one word
        new_words = [word for word in words if random.random() > p]
        return " ".join(new_words) if new_words else random.choice(words)
    
    def synonym_replacement(self, text, p=0.2):
        """Replace words in text with their synonyms with a given probability"""
        words = text.split()
        new_words = []

        for word in words:
            if random.random() < p:  # With probability p, replace the word
                synonyms = wordnet.synsets(word)
                if synonyms:
                    # Replace with the first synonym found
                    synonym = synonyms[0].lemmas()[0].name()
                    new_words.append(synonym)
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        return " ".join(new_words)

    
    def remove_chinese_and_punctuation(self, text):
        # 正则表达式：匹配所有中文字符和中文标点符号
        # \u4e00-\u9fa5：匹配常用汉字
        # \u3000-\u303F：匹配中文标点符号
        # \uff00-\uffef：匹配全角字符，包括一些标点
        return re.sub(r'[\u4e00-\u9fa5\u3000-\u303F\uff00-\uffef]', '', text)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # 查询这个图片的标签
        image_name = os.path.basename(image_path)
        label = self.label_dict[image_name]
        # image duqu 
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        transformed_image = self.transform(image)


        # 下面处理文本
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )

        return {
            'transformed_image': transformed_image,
            'label': int(label),
            # 'ori_text': self.texts[idx],
            'input_ids': encoding['input_ids'].squeeze(0),  # 去掉batch维度
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        

    
    def __len__(self):
        return len(self.image_paths)
        
class TextImageDatasetTrainISIC(Dataset):
    def __init__(self, json_data):
        self.image_path='/mnt/sda1/zzixuantang/classfier_convNext/data/02-isic2019/train-image/splitdata/train'

        df = pd.read_csv('/mnt/sda1/zzixuantang/classfier_convNext/data/02-isic2019/train.csv')
        class_dict = df.set_index('image_name')['diagnosis'].to_dict()
        class_func = {"MEL":0, "NV":1, "BCC":2, "AK":3, "BKL":4, "DF":5, "VASC":6, "SCC":7}
        self.label_dict = {}

        for imagename, enclass in class_dict.items():
            self.label_dict[imagename] = class_func[enclass]

        self.image_paths = [name for name in os.listdir(self.image_path)]




        self.text_path = json_data
        with open(self.text_path , 'r') as jsonf:
            text_dict = json.load(jsonf)
        text_dict2 = {}
        for k,v in text_dict.items():
            text_dict2[os.path.basename(k)]=v
        text_dict = text_dict2

        self.tokenizer = BertTokenizer.from_pretrained('./model/BERT_pretain')

        self.texts = []
        for path in self.image_paths:
            self.texts.append(self.remove_chinese_and_punctuation(text_dict[path]))
            

        
        # # ****************************************************************

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),          # 随机裁剪到224x224大小
            transforms.RandomHorizontalFlip(),          # 随机水平翻转
            transforms.RandomRotation(30),              # 随机旋转图像，范围为-30到30度
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机调整亮度、对比度、饱和度和色调
            transforms.RandomGrayscale(p=0.1),          # 以10%的概率将图像转换为灰度图像
            transforms.RandomVerticalFlip(p=0.1),       # 以10%的概率进行垂直翻转
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移图像
            transforms.ToTensor(),                      # 转换为Tensor格式
        ])

    def random_deletion(self, text, p=0.2):
        if len(text) <100:
            return text
        """Randomly delete words from the text"""
        words = text.split()
        if len(words) == 1:
            return text  # Prevent deletion if only one word
        new_words = [word for word in words if random.random() > p]
        return " ".join(new_words) if new_words else random.choice(words)
    
    def synonym_replacement(self, text, p=0.2):
        """Replace words in text with their synonyms with a given probability"""
        words = text.split()
        new_words = []

        for word in words:
            if random.random() < p:  # With probability p, replace the word
                synonyms = wordnet.synsets(word)
                if synonyms:
                    # Replace with the first synonym found
                    synonym = synonyms[0].lemmas()[0].name()
                    new_words.append(synonym)
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        return " ".join(new_words)

    
    def remove_chinese_and_punctuation(self, text):
        # 正则表达式：匹配所有中文字符和中文标点符号
        # \u4e00-\u9fa5：匹配常用汉字
        # \u3000-\u303F：匹配中文标点符号
        # \uff00-\uffef：匹配全角字符，包括一些标点
        return re.sub(r'[\u4e00-\u9fa5\u3000-\u303F\uff00-\uffef]', '', text)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # 查询这个图片的标签
        image_name = os.path.basename(image_path)
        label = self.label_dict[image_name.replace(".jpg","")]
        # image duqu 
        image = Image.open(self.image_path+"/"+image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        transformed_image = self.transform(image)


        # 下面处理文本
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )

        return {
            'transformed_image': transformed_image,
            'label': int(label),
            # 'ori_text': self.texts[idx],
            'input_ids': encoding['input_ids'].squeeze(0),  # 去掉batch维度
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        

    
    def __len__(self):
        return len(self.image_paths)
        



class TextImageDatasetTestISIC(Dataset):
    def __init__(self, json_data):
        self.image_path='/mnt/sda1/zzixuantang/classfier_convNext/data/02-isic2019/train-image/splitdata/train'

        df = pd.read_csv('/mnt/sda1/zzixuantang/classfier_convNext/data/02-isic2019/train.csv')
        class_dict = df.set_index('image_name')['diagnosis'].to_dict()
        class_func = {"MEL":0, "NV":1, "BCC":2, "AK":3, "BKL":4, "DF":5, "VASC":6, "SCC":7}
        self.label_dict = {}

        for imagename, enclass in class_dict.items():
            self.label_dict[imagename] = class_func[enclass]

        self.image_paths = [name for name in os.listdir(self.image_path)]




        self.text_path = json_data
        with open(self.text_path , 'r') as jsonf:
            text_dict = json.load(jsonf)
        text_dict2 = {}
        for k,v in text_dict.items():
            text_dict2[os.path.basename(k)]=v
        text_dict = text_dict2

        self.tokenizer = BertTokenizer.from_pretrained('./model/BERT_pretain')

        self.texts = []
        for path in self.image_paths:
            self.texts.append(self.remove_chinese_and_punctuation(text_dict[path]))
            

        
        # # ****************************************************************

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪到224x224大小
            # transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),              # 转换为Tensor格式
        ])

    
    def remove_chinese_and_punctuation(self, text):
        # 正则表达式：匹配所有中文字符和中文标点符号
        # \u4e00-\u9fa5：匹配常用汉字
        # \u3000-\u303F：匹配中文标点符号
        # \uff00-\uffef：匹配全角字符，包括一些标点
        return re.sub(r'[\u4e00-\u9fa5\u3000-\u303F\uff00-\uffef]', '', text)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # 查询这个图片的标签
        image_name = os.path.basename(image_path)
        label = self.label_dict[image_name.replace(".jpg","")]
        # image duqu 
        image = Image.open(self.image_path+"/"+image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        transformed_image = self.transform(image)


        # 下面处理文本
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )

        return {
            'transformed_image': transformed_image,
            'label': int(label),
            'input_ids': encoding['input_ids'].squeeze(0),  # 去掉batch维度
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        
    def __len__(self):
        return len(self.image_paths)
        





if __name__ == "__main__":
    db =TextImageDatasetTestISIC('/mnt/sda1/zzixuantang/classfier_convNext/data/02-isic2019/01-临床症状-4o-0217_modified.json')
    label = [0]*8
    for data in db:
        label[data["label"]]+=1
        print(label)
    # model = BertEncoder(num_labels=2)
    # for data in db:
    #     attention_mask = data['attention_mask']
    #     input_ids = data['input_ids']
    #     logits = model(input_ids.unsqueeze(dim=0), attention_mask.unsqueeze(dim=0))
    #     print(logits.shape)
        
import torch
import os
import numpy as np
from dataset.dataset import BaseDatasetTrain, BaseDatasetTest, TextImageDatasetTrain, TextImageDatasetTest
from torch.utils.data import DataLoader
from utils.loss import info_nce_loss

from transformers import ConvNextForImageClassification
import torch
import sys
import cv2
import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from matplotlib import pyplot as plt
from model.ourmodel import OurClassfier

# 训练过程
def train(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_data in tqdm(train_loader, desc="Training", ncols=100):
        batch_data = {key: value.to(device) for key, value in batch_data.items()}
        optimizer.zero_grad()
        outputs, ie, te = model(batch_data)
        loss = criterion(outputs, batch_data['label'])
        loss.backward()
        optimizer.step()

        scheduler.step()  # 更新学习率

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_data['label'].size(0)
        correct += predicted.eq(batch_data['label']).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def per_class_accuracy(confusion_matrix):
    class_accuracies = []
    for i in range(confusion_matrix.shape[0]):
        true_positive = confusion_matrix[i, i]
        false_positive = np.sum(confusion_matrix[:, i]) - true_positive
        false_negative = np.sum(confusion_matrix[i, :]) - true_positive
        # 类别准确率
        class_accuracy = true_positive / (true_positive + false_positive + false_negative) if true_positive + false_positive + false_negative > 0 else 0
        class_accuracies.append(class_accuracy)
    return class_accuracies


def test(model, test_loader, criterion, device, num_classes=7):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # 初始化混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing", ncols=100):
            batch_data = {key: value.to(device) for key, value in batch_data.items()}

            outputs, ie, te = model(batch_data)
            loss = criterion(outputs, batch_data['label'])

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_data['label'].size(0)
            correct += predicted.eq(batch_data['label']).sum().item()

            # 更新混淆矩阵
            for label, pred in zip(batch_data['label'], predicted):
                confusion_matrix[label.item(), pred.item()] += 1

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total
    class_accuracies=per_class_accuracy(confusion_matrix)

    return epoch_loss, epoch_acc, confusion_matrix, class_accuracies



def main():
    # 使用argparse获取超参数
    parser = argparse.ArgumentParser(description="Train a CNN model with cosine annealing learning rate")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam', help='Optimizer to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--train_data', type=str, default='./data', help='Path to training data')
    parser.add_argument('--test_data', type=str, default='./data', help='Path to test data')
    parser.add_argument('--ckpt', type=str, default='./workdir/1222_Image_textV2_exp_方案2attention_lr1e-5_onlytext_v3data/epoch_80_acc0.8710.pth', help='Path to test data')
    parser.add_argument('--save_epoch', type=int, default=20, help='Path to test data')
    parser.add_argument('--expname', type=str, default='1211_Image_textV2_exp_方案2attention_lr1e-5_onlyimage', help='Path to test data')
    parser.add_argument('--use_infonce_loss', type=bool, default=True, help='Path to test data')
    args = parser.parse_args()
    model_save_path = os.path.join('/mnt/sda1/zzixuantang/classfier_convNext/workdir', args.expname)
    os.makedirs(model_save_path, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    test_dataset = TextImageDatasetTest()
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化模型
    # model = ConvNextForImageClassification.from_pretrained('/mnt/sda1/zzixuantang/classfier_convNext/convnext-base-224')
    # model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=11)
    # model = model.cuda()
    model = OurClassfier(args, num_labels=7)
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt)
    model = model.cuda()

    # 定义损失函数
    if args.use_infonce_loss:
        criterion = infonce_loss
    else:
        criterion = nn.CrossEntropyLoss()

    # 选择优化器
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 余弦退火学习率调度器


    # 记录损失和准确率
    test_losses = []
    test_accuracies = []

    # 训练与测试


    # 测试
    test_loss, test_acc, martix, class_acc = test(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    print(f"Test Loss: {test_loss:.4f}, \nTest Accuracy: {test_acc:.4f}, \nclass_acc:{class_acc}")
    print(martix)

    

    # 每个epoch绘制loss和accuracy曲线
    # 绘制训练损失和测试损失
    # plot_loss(train_losses, test_losses, epoch + 1, model_save_path)
    # 绘制训练准确率和测试准确率
    # plot_accuracy(train_accuracies, test_accuracies, epoch + 1, model_save_path)


def plot_loss(train_losses, test_losses, epoch, save_path):
    # 创建一张空白图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, epoch + 1), test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    
    # 将图保存为图片
    loss_img_path = os.path.join(save_path, f"loss_curve_epoch_{epoch}.png")
    plt.savefig(loss_img_path)
    plt.close()



def plot_accuracy(train_accuracies, test_accuracies, epoch, save_path):
    # 创建一张空白图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 1), train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(range(1, epoch + 1), test_accuracies, label='Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    
    # 将图保存为图片
    acc_img_path = os.path.join(save_path, f"accuracy_curve_epoch_{epoch}.png")
    plt.savefig(acc_img_path)
    plt.close()


if __name__ == "__main__":
    main()


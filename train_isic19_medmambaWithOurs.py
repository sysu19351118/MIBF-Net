

import torch
import os
import numpy as np
from dataset.dataset import TextImageDatasetTestISIC, TextImageDatasetTrainISIC
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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
from tqdm import tqdm

from matplotlib import pyplot as plt
from model.medmambaWithOurs import MedMambaWithOur
from utils.loss import InfoNCELoss

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 训练过程
def train(model, train_loader, criterion, optimizer, scheduler, device, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    correct_image = 0
    correct_text = 0
    total = 0

    for batch_data in tqdm(train_loader, desc="Training", ncols=100):
        batch_data = {key: value.to(device) for key, value in batch_data.items()}
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = model.module.cal_loss(outputs, batch_data['label'])
        outputs = outputs['image_text']
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()  # 更新学习率

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        _, predicted_image = outputs.max(1)
        _, predicted_lable = outputs.max(1)
        total += batch_data['label'].size(0)
        correct += predicted.eq(batch_data['label']).sum().item()
        correct_image += predicted.eq(batch_data['label']).sum().item()
        correct_text += predicted.eq(batch_data['label']).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    epoch_acc_image = correct_image / total
    epoch_acc_text = correct_text / total
    return epoch_loss, epoch_acc, epoch_acc_image, epoch_acc_text


# 测试过程
def test(model, test_loader, criterion, device, num_classes=8):
    model.eval()
    running_loss = 0.0
    correct = 0
    correct_image = 0
    correct_text = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing", ncols=100):
            batch_data = {key: value.to(device) for key, value in batch_data.items()}

            outputs = model(batch_data)
            loss = model.module.cal_loss(outputs, batch_data['label'])
            outputs_cross = outputs['image_text']
            outputs_image = outputs['image']
            outputs_text = outputs['text']

            running_loss += loss.item()
            _, predicted = outputs_cross.max(1)
            _, predicted_image = outputs_image.max(1)
            _, predicted_text = outputs_text.max(1)
            total += batch_data['label'].size(0)
            correct += predicted.eq(batch_data['label']).sum().item()
            correct_image += predicted_image.eq(batch_data['label']).sum().item()
            correct_text += predicted_text.eq(batch_data['label']).sum().item()

            all_preds.extend(predicted.cpu().numpy())  # 收集预测结果
            all_labels.extend(batch_data['label'].cpu().numpy())  # 收集真实标签

    # 计算损失和准确率
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total
    epoch_acc_image = correct_image / total
    epoch_acc_text = correct_text / total

    # 计算每个类别的精确率、召回率和F1值
    precision_list = precision_score(all_labels, all_preds, average=None, labels=list(range(num_classes)))
    recall_list = recall_score(all_labels, all_preds, average=None, labels=list(range(num_classes)))
    f1_list = f1_score(all_labels, all_preds, average=None, labels=list(range(num_classes)))

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    return epoch_loss, epoch_acc, epoch_acc_image, epoch_acc_text, precision_list, recall_list, f1_list, conf_matrix


def main():
    # 使用argparse获取超参数
    parser = argparse.ArgumentParser(description="Train a CNN model with cosine annealing learning rate")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam', help='Optimizer to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--train_data', type=str, default='', help='Path to training data')
    parser.add_argument('--train_json_path', type=str, default='/mnt/sda1/zzixuantang/classfier_convNext/data/02-isic2019/01-临床症状-4o-0217_modified.json', help='Path to training data')
    parser.add_argument('--test_json_path', type=str, default='/mnt/sda1/zzixuantang/classfier_convNext/data/02-isic2019/01-临床症状-4o-0217_modified.json', help='Path to training data')
    parser.add_argument('--test_data', type=str, default='./data', help='Path to test data')
    parser.add_argument('--loss_type', type=str, default='KL_loss', help='Path to test data')
    parser.add_argument('--save_epoch', type=int, default=20, help='Path to test data')
    parser.add_argument('--expname', type=str, default='test', help='Path to test data')
    parser.add_argument('--use_infonce_loss', type=bool, default=True, help='Path to test data')
    

    # DDP parameters
    parser.add_argument('--local_rank', type=int, help='local_rank for distributed training')
    
    args = parser.parse_args()

    # Set up distributed training
    dist.init_process_group(backend='nccl')  # Initialize the process group for DDP
    local_rank = int(os.environ['LOCAL_RANK'])
    args.local_rank = local_rank
    device = torch.device(f"cuda:{local_rank}")
    model_save_path = os.path.join('./workdir', args.expname)
    os.makedirs(model_save_path, exist_ok=True)



    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 简单的归一化
    ])

    # 定义数据加载器
    train_dataset = TextImageDatasetTrainISIC(args.train_json_path)
    train_sampler = DistributedSampler(train_dataset)  # Use DistributedSampler for distributed training
    train_loader = DataLoader(
        train_dataset,  # 数据集实例
        batch_size=args.batch_size,  # 每批次加载的样本数量
        shuffle=False,  # Shuffle should be False when using DistributedSampler
        num_workers=32,  # 加载数据时使用的线程数
        pin_memory=True,  # 是否将数据预加载到GPU内存，适用于GPU训练
        drop_last=True,  # 如果数据集大小不能被 batch_size 整除，是否丢弃最后不完整的批次
        sampler=train_sampler  # Use sampler for distributed training
    )
    
    test_dataset = TextImageDatasetTestISIC(args.test_json_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MedMambaWithOur(num_labels=8, loss_class=args.loss_type)
    model = model.to(device)

    # Wrap the model with DDP
    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    # 定义损失函数
    if args.use_infonce_loss:
        criterion = InfoNCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # 选择优化器
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 余弦退火学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 记录损失和准确率
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # 训练与测试
    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')

        # Update sampler for each epoch
        train_sampler.set_epoch(epoch)

        # 训练
        train_loss, train_acc, train_acc_img, train_acc_text = train(model, train_loader, criterion, optimizer, scheduler, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # 测试
        test_loss, test_acc, test_acc_image, test_acc_text, precision_list, recall_list, f1_list, conf_matrix = test(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        # 记录实验结果

        with open(f"{model_save_path}/test_result.pth", 'a') as f:
            f.write(f"\n\nepoch: {epoch} \nTest Loss: {test_loss:.4f},\nTest Accuracy: {test_acc:.4f} \nTest Accuracy Image: {test_acc_image:.4f} \nTest Accuracy Text: {test_acc_text:.4f}\n精准率:{precision_list}\n召回率:{recall_list}\nf1分数:{f1_list},\n混淆矩阵:\n{conf_matrix}")
        if epoch % args.save_epoch == 0 and epoch != 0:
            torch.save(model.state_dict(), f"{model_save_path}/epoch_{epoch}_acc{test_acc:.4f}.pth")

            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f"{model_save_path}/best_model_acc{test_acc:.4f}.pth")

        # 每个epoch绘制loss和accuracy曲线
        # 绘制训练损失和测试损失
        plot_loss(train_losses, test_losses, epoch + 1, model_save_path)
        # 绘制训练准确率和测试准确率
        plot_accuracy(train_accuracies, test_accuracies, epoch + 1, model_save_path)



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
    loss_img_path = os.path.join(save_path, f"loss_curve_epoch.png")
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
    acc_img_path = os.path.join(save_path, f"accuracy_curve_epoch.png")
    plt.savefig(acc_img_path)
    plt.close()


if __name__ == "__main__":
    main()


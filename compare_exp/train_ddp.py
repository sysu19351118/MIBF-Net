
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import argparse
from PIL import Image
from PIL import Image
import timm
import torch.nn as nn
from torchvision import transforms
from transformers import ConvNextForImageClassification
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, f1_score
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class ConvNextWithClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(ConvNextWithClassifier, self).__init__()
        self.backbone = ConvNextForImageClassification.from_pretrained('/mnt/sda1/zzixuantang/classfier_convNext/convnext-base-224').convnext
        self.classifier = nn.Linear(self.backbone.config.hidden_sizes[-1], num_classes)
    
    def forward(self, x):
        # 通过 backbone 提取特征
        features = self.backbone(x).last_hidden_state
        # 全局平均池化
        pooled_features = features.mean(dim=[2, 3])
        # 通过分类器进行分类
        logits = self.classifier(pooled_features)
        return logits


class MyDataset(Dataset):
    def __init__(self, image_path, label_dict, mode):
        
        self.image_paths = [os.path.join(image_path, image_name) for image_name in os.listdir(image_path)]
        
        data_transform = {'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        self.transform = data_transform[mode]
        self.label_dict = label_dict


    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image_data = self.transform(image)
        try:
            label = int(self.label_dict[os.path.basename(image_path).replace(".jpg","")])
        except:
            label = int(self.label_dict[os.path.basename(image_path)])
        return image_data, label
    
    def __len__(self):
        return len(self.image_paths)

def getisic():
    df = pd.read_csv('/mnt/sda1/zzixuantang/classfier_convNext/data/02-isic2019/train.csv')
    class_dict = df.set_index('image_name')['diagnosis'].to_dict()
    class_func = {"MEL":0, "NV":1, "BCC":2, "AK":3, "BKL":4, "DF":5, "VASC":6, "SCC":7}
    label_dict = {}
    for imagename, enclass in class_dict.items():
        label_dict[imagename] = class_func[enclass]
    traindataset = MyDataset("/mnt/sda1/zzixuantang/classfier_convNext/data/02-isic2019/train-image/splitdata/train", label_dict, "train")
    testdataset = MyDataset("/mnt/sda1/zzixuantang/classfier_convNext/data/02-isic2019/train-image/splitdata/test", label_dict, "test")
    return traindataset, testdataset

def getham():
    traindf = pd.read_csv('/mnt/sda1/zzixuantang/classfier_convNext/data/00-HAM10000/label/split_data_1_fold_train.csv')
    train_label_dict = traindf.set_index('image_train')['label_train'].to_dict()


    testdf = pd.read_csv('/mnt/sda1/zzixuantang/classfier_convNext/data/00-HAM10000/label/split_data_1_fold_test.csv')
    test_label_dict = testdf.set_index('image_test')['label_test'].to_dict()
    label_dict = {}
    for k,v in train_label_dict.items():
        label_dict[k] = v
    for k,v in test_label_dict.items():
        label_dict[k] = v
        
    traindataset = MyDataset("/mnt/sda1/zzixuantang/classfier_convNext/data/00-HAM10000/train/images", label_dict, "train")
    testdataset = MyDataset("/mnt/sda1/zzixuantang/classfier_convNext/data/00-HAM10000/test/images", label_dict, "test")
    return traindataset, testdataset



import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
import numpy as np

# 训练函数
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    train_loader.sampler.set_epoch(epoch)  # 打乱数据
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# 测试函数
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import torch

def test(model, test_loader, device, num_classes=7, output_file="test_results.txt"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算每个类别的准确率
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    for pred, label in zip(all_preds, all_labels):
        class_correct[label] += int(pred == label)
        class_total[label] += 1

    class_accuracy = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]

    # 计算 F1-macro 和 F1-micro
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')

    # 生成分类报告
    classification_rep = classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(num_classes)], digits=4 )

    # 将结果写入文件
    with open(output_file, "a") as f:
        f.write("\n\n\n")
        f.write("Class-wise Accuracy:\n")
        for i, acc in enumerate(class_accuracy):
            f.write(f"Class {i}: {acc:.4f}\n")

        f.write(f"\nF1-macro: {f1_macro:.4f}\n")
        f.write(f"F1-micro: {f1_micro:.4f}\n")

        f.write("\nClassification Report:\n")
        f.write(classification_rep)

    print(f"Test results saved to {output_file}")




def main(model_name, datasetname, save_path):
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # 获取数据集和数据加载器
    if datasetname=="ham":
        train_dataset, test_dataset = getham()
    else:
        train_dataset, test_dataset = getisic()
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, sampler=test_sampler, num_workers=12, pin_memory=True)

    # 定义模型
    import sys
    sys.path.append('/mnt/sda1/zzixuantang/classfier_convNext/model')
    from hifuse import Hifuse
    sys.path.append('/home/zzixuantang/playground/spa/SPA-main')
    from SPAnet import spa_tiny_224
    # sys.path.append('/mnt/sda1/zzixuantang/classfier_convNext/model')
    # from medmambaWithOurs import MedMamba

    if model_name == "resnet":
        model = models.resnet50(pretrained=True)
        if datasetname=="ham":
            model.fc = nn.Linear(model.fc.in_features, 7)  # 假设有8个类别
        else:
            model.fc = nn.Linear(model.fc.in_features, 8)  # 假设有8个类别
    elif model_name == "convnext":
        if datasetname=='ham':
            model = ConvNextWithClassifier(7)
        else:
            model = ConvNextWithClassifier(8)
    elif model_name=="hifuse":
        if datasetname=='ham':
            model = Hifuse(7)
        else:
            model = Hifuse(8)
    elif model_name=="spa":
        if datasetname=='ham':
            model = spa_tiny_224(7)
        else:
            model = spa_tiny_224(8)

    elif model_name=="davit":
        model = timm.create_model('davit_base.msft_in1k', pretrained=False)
        checkpoint = torch.load('/mnt/sda1/zzixuantang/davit/pytorch_model.bin')
        model.load_state_dict(checkpoint)  # 根据权重文件的具体结构调整
        if datasetname=='ham':
            model.head.fc = nn.Linear(1024, 7)
        else:
            model.head.fc = nn.Linear(1024, 8)
    elif model_name=="medmamba":
        model = MedMamba(7) if datasetname=='ham' else  MedMamba(8)

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # 训练和验证循环
    num_epochs = 100
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        if local_rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        if local_rank == 0:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{save_path}/best_model.pth")
        if (epoch+1)%5==0:
            if local_rank == 0:
                print(f"Best Validation Accuracy: {best_val_acc:.4f}%")
                print("\nTesting the best model...")
                state_dict = torch.load("best_model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
                numclass=7 if datasetname=="ham" else 8
                test(model, test_loader, device, numclass, output_file=f"{save_path}/test_result.txt")

    # 清理分布式环境
    dist.destroy_process_group()

if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    
    parser = argparse.ArgumentParser(description="Train a model with specified network and dataset.")
    
    # 添加命令行参数
    parser.add_argument("--netname", type=str, required=True, help="Name of the network (e.g., convnext)")
    parser.add_argument("--dataset", type=str, required=True, help="ham or isic")
    parser.add_argument("--exp_name", type=str, required=True, help="ham or isic")
    
    # 解析命令行参数
    args = parser.parse_args()
    save_path = f"/mnt/sda1/zzixuantang/classfier_convNext/compare_exp/workdir/{args.exp_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 调用 main 函数并传入参数
    main(args.netname, args.dataset, save_path)
    


        

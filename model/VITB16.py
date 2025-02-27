import torch
from transformers import ViTModel, ViTConfig
from PIL import Image
import torchvision.transforms as transforms
from torch import nn


# 加载预训练的ViT模型配置

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class Adapter(nn.Module):
    def __init__(self, input_dim, adapter_dim):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_dim, input_dim)

    def forward(self, x):
        down = self.down_project(x)
        activated = self.activation(down)
        up = self.up_project(activated)
        return x + up  # Residual connection

class VITB16_encoder(nn.Module):
    def __init__(self, adapter_dim=64):  # adapter_dim is the dimension of the adapter bottleneck
        super(VITB16_encoder, self).__init__()
        model_name = "/mnt/sda1/zzixuantang/classfier_convNext/model/VITB16_pretrain"
        config = ViTConfig.from_pretrained(model_name)
        self.model = ViTModel(config)
        
        # Add adapters to each layer
        self.adapters = nn.ModuleList()
        for i, layer in enumerate(self.model.encoder.layer):
            adapter = Adapter(config.hidden_size, adapter_dim)
            self.adapters.append(adapter)

    def forward(self, x):
        # Forward pass through the embedding layer
        x = self.model.embeddings(x)
        
        # Forward pass through each encoder layer and adapter
        for i, layer in enumerate(self.model.encoder.layer):
            x = layer(x)[0]  # Get the hidden states from the layer
            x = self.adapters[i](x)  # Pass through the adapter
        
        # Forward pass through the final layer norm
        x = self.model.layernorm(x)
        
        return x



if __name__ == "__main__":
    # # 定义图像转换（通常ViT使用的预处理）
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),  # 可能需要调整图像大小
    #     transforms.ToTensor(),          # 转换为Tensor
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet标准
    # ])
    # 前向传播，获取隐层特征
    model = VITB16_encoder()
    # print(model)
    input_tensor = torch.randn(1, 3, 224, 224)  # Example input
    output = model(input_tensor)
    print(output[:,0,:].shape)

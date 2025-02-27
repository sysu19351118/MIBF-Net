from .BERT import BertEncoder
import torch
from torch import nn
from .VITB16 import VITB16_encoder
from transformers import ConvNextForImageClassification
import sys
from torchvision import models, transforms

class OurClassfier(nn.Module):
    def __init__(self, num_labels=2):  # num_labels 代表分类的数量
        super(OurClassfier, self).__init__()
        self.text_encoder = BertEncoder()
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 768)  # 假设有8个类别
        self.loss = nn.CrossEntropyLoss()
        
        self.image_encoder = model
        # # **************************   方案1    拼接    ****************************
        # self.fc = nn.Sequential(
        #     nn.Linear(768 * 2, 512),  # 假设拼接后的维度为 768*2
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_labels)  
        # )
        # # ************************************************************************

        # ************************** 方案2 attention   ****************************
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_labels)
        )
        # ************************************************************************

    def forward(self, batch_data):
        
        text_embedding = self.text_encoder(batch_data['input_ids'], batch_data['attention_mask'])
        image_embedding = self.image_encoder(batch_data['transformed_image'])

        # # **************************   方案1    拼接    ****************************
        # fused_features = torch.cat((text_embedding, image_embedding), dim=-1)  # 拼接
        # output = self.fc(fused_features)
        # # ************************************************************************

        fused_features = torch.stack([image_embedding, image_embedding], dim=0)
        # print(fused_features.shape)
        attn_output, _ = self.attention(fused_features, fused_features, fused_features)
        fused_features = attn_output.mean(dim=0)  # 对两个模态的特征进行平均
        output = self.fc(fused_features)
        
        return output

    def cal_loss(self, output, batch):
        return self.loss(output, batch)
        


class BaselineSimpleConcat(nn.Module):
    # 对比实验，证明比简单的concat有效
    def __init__(self, args, num_labels=2):  # num_labels 代表分类的数量
        super(BaselineSimpleConcat, self).__init__()
        self.text_encoder = BertEncoder()
        image_encoder_model = ConvNextForImageClassification.from_pretrained('./convnext-base-224')
        self.image_encoder = image_encoder_model.convnext
        self.conv = nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=1)
        self.textbased_cross_attention = CrossAttention(dim=768)
        self.imagbased_cross_attention = CrossAttention(dim=768)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_labels)

    def forward(self, batch_data):
        
        text_embedding = self.text_encoder(batch_data['input_ids'], batch_data['attention_mask'])
        image_embedding = self.image_encoder(batch_data['transformed_image']).last_hidden_state
        image_embedding_reduced = self.conv(image_embedding)
        text_embedding_expanded = text_embedding
        image_embedding_reduced = self.avg_pool(image_embedding_reduced).view(batch_data['transformed_image'].shape[0], 768)

        output = self.fc(text_embedding_expanded+image_embedding_reduced)

        return output

class BaselineOnlytext(nn.Module):
    # 这个版本分别使用image和text的embedding作为kqv 然后再将输出进行融合
    def __init__(self, args, num_labels=2):  # num_labels 代表分类的数量
        super(BaselineOnlytext, self).__init__()
        self.text_encoder = BertEncoder()
        self.fc = nn.Linear(768, num_labels)

    def forward(self, batch_data):
        
        text_embedding = self.text_encoder(batch_data['input_ids'], batch_data['attention_mask'])
        output = self.fc(text_embedding)

        return output

class BaselineOnlyimage(nn.Module):
    # 这个版本分别使用image和text的embedding作为kqv 然后再将输出进行融合
    def __init__(self, args, num_labels=2):  # num_labels 代表分类的数量
        super(BaselineOnlyimage, self).__init__()
        self.image_encoder = ConvNextForImageClassification.from_pretrained('/mnt/sda1/zzixuantang/classfier_convNext/convnext-base-224')
        self.image_encoder.classifier = torch.nn.Linear(in_features=self.image_encoder.classifier.in_features, out_features=num_labels)

    def forward(self, batch_data):
        
       
        output = self.image_encoder(batch_data['transformed_image']).logits

        return output

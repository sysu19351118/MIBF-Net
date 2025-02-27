from .BERT import BertEncoder
from transformers import ConvNextForImageClassification
from .ourmodel import MultiHeadCrossAttention_v2, compute_kl_divergence, SelfAttention
import torch.nn as nn
import sys
import torch
import torch.nn.functional as F

import sys
sys.path.append('/home/zzixuantang/playground/pip-net/PIPNet-main/pipnet')
from pipnet import getnetwork

class PipnetWithOur(nn.Module):
    """
    这个版本分别使用 image 和 text 的 embedding 作为 kqv 然后再将输出进行融合。
    """
    def __init__(self, num_labels=2, loss_class="textimage_loss"):

        super(PipnetWithOur, self).__init__()
        self.text_encoder = BertEncoder()
        
        self.image_encoder = getnetwork()
        # self.conv = nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=1)
        self.textbased_cross_attention = MultiHeadCrossAttention_v2(dim=768,num_heads=1)
        self.imagbased_cross_attention = MultiHeadCrossAttention_v2(dim=768,num_heads=1)
        self.I2Iattention = SelfAttention(input_dim=768)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768 * 2, num_labels)
        self.fc_image = self._build_mlp(768, num_labels)
        self.fc_text = self._build_mlp(768, num_labels)
        self.loss_class = loss_class
        self.loss = nn.CrossEntropyLoss()

    def _build_mlp(self, input_dim, num_labels):
        return nn.Sequential(
            nn.Flatten(start_dim=1),  # 展平输入，忽略 batch 维度
            nn.Linear(input_dim, 512),  # 第一个全连接层
            nn.ReLU(),  # 激活函数
            nn.Linear(512, num_labels)  # 输出层
        )

    def forward(self, batch_data):
        text_embedding = self.text_encoder(batch_data['input_ids'], batch_data['attention_mask'])
        _,image_embedding,_ = self.image_encoder(batch_data['transformed_image'])
        image_embedding_reduced = image_embedding
        text_embedding_expanded = text_embedding.unsqueeze(dim=1)
        
        # image_embedding_reduced = self.I2Iattention(image_embedding_reduced)
        # image_embedding_pooled = self.avg_pool(image_embedding_reduced).view(batch_data['transformed_image'].shape[0], 768).unsqueeze(dim=1)
        image_embedding_pooled = image_embedding_reduced.unsqueeze(dim=1)
        text_fused_features = self.textbased_cross_attention(image_embedding_pooled, text_embedding_expanded)
        pooled_features_1 = text_fused_features.view(batch_data['transformed_image'].shape[0], 768)
        

        imag_fused_features = self.imagbased_cross_attention(text_embedding_expanded, image_embedding_pooled)
        pooled_features_2 = imag_fused_features.squeeze()

        output = {}
        output['image_text'] = self.fc(torch.cat([pooled_features_1, pooled_features_2], dim=1))
        output['text'] = self.fc_text(text_fused_features)
        output['image'] = self.fc_image(imag_fused_features)

        return output

    def cal_loss(self, output, batch):
        if self.loss_class == "textimage_loss":
            return self.loss(output['image_text'], batch)
        elif self.loss_class == 'text_image_textimage_loss':
            return self.loss(output['image'], batch)+self.loss(output['text'], batch)+self.loss(output['image_text'], batch)
        elif self.loss_class == "KL_loss":
            return self.compute_kl_loss(output, batch)


    def compute_kl_loss(self, output, batch):
        image_logits = output['image']
        text_logits = output['text']
        image_text_logits = output['image_text']
        image_prob = F.softmax(image_logits, dim=-1)
        text_prob = F.softmax(text_logits, dim=-1)

        kl = (compute_kl_divergence(image_prob, text_prob) + compute_kl_divergence(text_prob, image_prob))/2  # KL(image || image_text)
        image_loss = F.cross_entropy(image_logits, batch)
        text_loss = F.cross_entropy(text_logits, batch)
        image_text_loss = F.cross_entropy(image_text_logits, batch)
        weight_factor = torch.exp(kl)  # 通过指数放大KL散度大的样本的影响
        weighted_image_text_loss = torch.mean((weight_factor)* image_text_loss)
        total_loss = 0.3*image_loss + 0.6*text_loss + 1.1*weighted_image_text_loss

        return total_loss


if __name__ == "__main__":
    net = PipnetWithOur()
    a = torch.randn(4,3,224,224)
    
    print(net(a))
    

import sys
sys.path.append('/mnt/sda1/zzixuantang/classfier_convNext/model')
from BERT import BertEncoder
import torch
from torch import nn
from VITB16 import VITB16_encoder
from transformers import ConvNextForImageClassification
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        # Query, Key, Value projections
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Perform self-attention on the input tensor.
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            Tensor of the same shape as input with self-attention applied.
        """
        batch_size, channels, height, width = x.size()
        flattened = x.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, H*W, C)

        # Compute Query, Key, Value
        query = self.query(flattened)  # (batch_size, H*W, C)
        key = self.key(flattened).permute(0, 2, 1)  # (batch_size, C, H*W)
        value = self.value(flattened)  # (batch_size, H*W, C)

        # Compute attention scores
        attention_scores = torch.bmm(query, key)  # (batch_size, H*W, H*W)
        attention_scores = self.softmax(attention_scores / (channels ** 0.5))  # Scale and apply softmax

        # Compute attended values
        attended_values = torch.bmm(attention_scores, value)  # (batch_size, H*W, C)
        attended_values = attended_values.permute(0, 2, 1).view(batch_size, channels, height, width)

        return attended_values


def compute_kl_divergence(p, q):
    # p 和 q 都是经过 softmax 转换的概率分布
    return torch.sum(p * torch.log(p / (q + 1e-10)), dim=-1)

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.key_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.value_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        query = self.query_conv(x)
        key = self.key_conv(y)
        value = self.value_conv(y)
        
        attention = self.softmax(torch.matmul(query.view(query.size(0), query.size(1), -1).permute(0, 2, 1),
                                              key.view(key.size(0), key.size(1), -1)))
        out = torch.matmul(attention, value.view(value.size(0), value.size(1), -1).permute(0, 2, 1))
        out = out.permute(0, 2, 1).view(x.size())
        return out


# class CrossAttention_v2(nn.Module):
#     def __init__(self, dim):
#         super(CrossAttention_v2, self).__init__()
#         self.query_conv = nn.Conv2d(dim, dim, kernel_size=1)
#         self.key_conv = nn.Conv2d(2*dim, dim, kernel_size=1)
#         self.value_conv = nn.Conv2d(2*dim, dim, kernel_size=1)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, y):
#         y = torch.cat([x, y], axis=1)
#         query = self.query_conv(x)
#         key = self.key_conv(y)
#         value = self.value_conv(y)

        
        
#         attention = self.softmax(torch.matmul(query.view(query.size(0), query.size(1), -1).permute(0, 2, 1),
#                                               key.view(key.size(0), key.size(1), -1)))
#         out = torch.matmul(attention, value.view(value.size(0), value.size(1), -1).permute(0, 2, 1))
#         out = out.permute(0, 2, 1).view(x.size())
#         out = out+x

#         return out

class CrossAttention_v2(nn.Module):
    def __init__(self, dim):
        super(CrossAttention_v2, self).__init__()
        self.toK_x = nn.Linear(dim, dim)
        self.toQ_x = nn.Linear(dim, dim)
        self.toV_x = nn.Linear(dim, dim)
        self.toK_y = nn.Linear(dim, dim)
        self.toV_y = nn.Linear(dim, dim)

    def forward(self, x, y):
        # 将 x 转换为 Kx, Qx, Vx
        Kx = self.toK_x(x)
        Qx = self.toQ_x(x)
        Vx = self.toV_x(x)

        # 将 y 转换为 Ky, Vy
        Ky = self.toK_y(y)
        Vy = self.toV_y(y)

        # 在序列长度维度（dim=1）上拼接 K 和 V
        Kcat = torch.cat([Kx, Ky], dim=1)
        Vcat = torch.cat([Vx, Vy], dim=1)
        attention_scores = torch.matmul(Qx, Kcat.transpose(-2, -1))
        attention_scores = attention_scores / (Kcat.size(-1) ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, Vcat)

        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention_v2(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadCrossAttention_v2, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # Linear layers for x
        self.toK_x = nn.Linear(dim, dim)
        self.toQ_x = nn.Linear(dim, dim)
        self.toV_x = nn.Linear(dim, dim)

        # Linear layers for y
        self.toK_y = nn.Linear(dim, dim)
        self.toV_y = nn.Linear(dim, dim)

        # Output linear layer
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, y):
        batch_size, seq_len_x, _ = x.shape
        _, seq_len_y, _ = y.shape

        # 将 x 转换为 Kx, Qx, Vx
        Kx = self.toK_x(x)
        Qx = self.toQ_x(x)
        Vx = self.toV_x(x)

        # 将 y 转换为 Ky, Vy
        Ky = self.toK_y(y)
        Vy = self.toV_y(y)

        # 将 Kx, Qx, Vx, Ky, Vy 分割成多个头
        Kx = Kx.view(batch_size, seq_len_x, self.num_heads, self.head_dim).transpose(1, 2)
        Qx = Qx.view(batch_size, seq_len_x, self.num_heads, self.head_dim).transpose(1, 2)
        Vx = Vx.view(batch_size, seq_len_x, self.num_heads, self.head_dim).transpose(1, 2)
        Ky = Ky.view(batch_size, seq_len_y, self.num_heads, self.head_dim).transpose(1, 2)
        Vy = Vy.view(batch_size, seq_len_y, self.num_heads, self.head_dim).transpose(1, 2)

        # 在序列长度维度（dim=2）上拼接 K 和 V
        Kcat = torch.cat([Kx, Ky], dim=2)
        Vcat = torch.cat([Vx, Vy], dim=2)
        attention_scores = torch.matmul(Qx, Kcat.transpose(-2, -1))
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, Vcat)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_x, self.dim)
        output = self.to_out(output)

        return output


class OurClassfierConvnext(nn.Module):
    def __init__(self, args, num_labels=2):  # num_labels 代表分类的数量
        super(OurClassfierConvnext, self).__init__()
        self.text_encoder = BertEncoder()
        image_encoder_model = ConvNextForImageClassification.from_pretrained('./convnext-base-224')
        self.image_encoder = image_encoder_model.convnext
        self.conv = nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=1)
        self.cross_attention = CrossAttention(dim=768)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_labels)

    def forward(self, batch_data):
        
        text_embedding = self.text_encoder(batch_data['input_ids'], batch_data['attention_mask'])
        image_embedding = self.image_encoder(batch_data['transformed_image']).last_hidden_state
        image_embedding_reduced = self.conv(image_embedding)
        text_embedding_expanded = text_embedding.unsqueeze(-1).unsqueeze(-1)
        fused_features = self.cross_attention(image_embedding_reduced, text_embedding_expanded)
        pooled_features = self.avg_pool(fused_features).view(batch_data['transformed_image'].shape[0], 768)
        output = self.fc(pooled_features)
        return output

class OurClassfierConvnextV2(nn.Module):
    # 这个版本分别使用image和text的embedding作为kqv 然后再将输出进行融合
    def __init__(self, args, num_labels=2):  # num_labels 代表分类的数量
        super(OurClassfierConvnextV2, self).__init__()
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
        text_embedding_expanded = text_embedding.unsqueeze(-1).unsqueeze(-1)
        text_fused_features = self.textbased_cross_attention(image_embedding_reduced, text_embedding_expanded)
        pooled_features_1 = self.avg_pool(text_fused_features).view(batch_data['transformed_image'].shape[0], 768)

        imag_fused_features = self.imagbased_cross_attention(text_embedding_expanded,image_embedding_reduced)
        pooled_features_2 = self.avg_pool(imag_fused_features).view(batch_data['transformed_image'].shape[0], 768)
        output = self.fc(pooled_features_1+pooled_features_2)
        return output

class OurClassfierConvnextV3(nn.Module):
    # 这个版本分别使用image和text的embedding作为kqv 然后再将输出进行融合
    def __init__(self, args, num_labels=2):  # num_labels 代表分类的数量
        super(OurClassfierConvnextV3, self).__init__()
        self.text_encoder = BertEncoder()
        image_encoder_model = ConvNextForImageClassification.from_pretrained('./convnext-base-224')
        self.image_encoder = image_encoder_model.convnext
        self.conv = nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=1)
        self.textbased_cross_attention = CrossAttention(dim=768)
        self.imagbased_cross_attention = CrossAttention(dim=768)
        self.I2Iattention = SelfAttention(input_dim=768)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_labels)



    def forward(self, batch_data):
        
        text_embedding = self.text_encoder(batch_data['input_ids'], batch_data['attention_mask'])
        image_embedding = self.image_encoder(batch_data['transformed_image']).last_hidden_state
        image_embedding_reduced = self.conv(image_embedding)
        text_embedding_expanded = text_embedding.unsqueeze(-1).unsqueeze(-1)

        # 只有image经过attention
        image_embedding_reduced = self.I2Iattention(image_embedding_reduced)

        text_fused_features = self.textbased_cross_attention(image_embedding_reduced, text_embedding_expanded)
        pooled_features_1 = self.avg_pool(text_fused_features).view(batch_data['transformed_image'].shape[0], 768)

        imag_fused_features = self.imagbased_cross_attention(text_embedding_expanded,image_embedding_reduced)
        pooled_features_2 = self.avg_pool(imag_fused_features).view(batch_data['transformed_image'].shape[0], 768)
        output = self.fc(pooled_features_1+pooled_features_2)

        # loss
        te = text_embedding
        ie = self.avg_pool2(image_embedding_reduced)
        return output, ie.squeeze(), te

class OurClassfierConvnextV4(nn.Module):
    # 这个版本分别使用image和text的embedding作为kqv 然后再将输出进行融合
    def __init__(self, args, num_labels=2, loss_class="textimage_loss"):  # num_labels 代表分类的数量
        super(OurClassfierConvnextV4, self).__init__()
        self.text_encoder = BertEncoder()
        image_encoder_model = ConvNextForImageClassification.from_pretrained('./convnext-base-224')
        self.image_encoder = image_encoder_model.convnext
        self.conv = nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=1)
        self.textbased_cross_attention = CrossAttention(dim=768)
        self.imagbased_cross_attention = CrossAttention(dim=768)
        self.I2Iattention = SelfAttention(input_dim=768)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768*2, num_labels)
        self.fc_image = nn.Sequential(
            nn.Flatten(),  # 展平输入
            nn.Linear(768, 512),  # 第一个全连接层
            nn.ReLU(),  # 激活函数
            nn.Linear(512, num_labels)  # 输出层，7个类别
        )
        
        # 第二个MLP处理torch.Size([32, 768, 7, 7])的输入
        self.fc_text = nn.Sequential(
            nn.Flatten(start_dim=1),  # 展平输入，忽略batch维度
            nn.Linear(768 * 7 * 7, 512),  # 第一个全连接层
            nn.ReLU(),  # 激活函数
            nn.Linear(512, num_labels)  # 输出层，7个类别
        )
        self.loss_class = loss_class
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch_data):
        
        text_embedding = self.text_encoder(batch_data['input_ids'], batch_data['attention_mask'])
        image_embedding = self.image_encoder(batch_data['transformed_image']).last_hidden_state
        image_embedding_reduced = self.conv(image_embedding)
        text_embedding_expanded = text_embedding.unsqueeze(-1).unsqueeze(-1)

        # 只有image经过attention
        image_embedding_reduced = self.I2Iattention(image_embedding_reduced)

        text_fused_features = self.textbased_cross_attention(image_embedding_reduced, text_embedding_expanded)
        pooled_features_1 = self.avg_pool(text_fused_features).view(batch_data['transformed_image'].shape[0], 768)


        imag_fused_features = self.imagbased_cross_attention(text_embedding_expanded,image_embedding_reduced)
        pooled_features_2 = self.avg_pool(imag_fused_features).view(batch_data['transformed_image'].shape[0], 768)
        output = {}
        output['image_text'] = self.fc(torch.cat([pooled_features_1,pooled_features_2], axis=1))
        # loss
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
        # 提取输出的logits
        image_logits = output['image']
        text_logits = output['text']
        image_text_logits = output['image_text']

        # 使用softmax将logits转换为概率分布
        image_prob = F.softmax(image_logits, dim=-1)
        text_prob = F.softmax(text_logits, dim=-1)

        image_text_prob = F.softmax(image_text_logits, dim=-1)

        # 计算KL散度
        kl = compute_kl_divergence(image_prob, text_prob)  # KL(image || image_text)
        
        # 计算单模态的分类损失
        image_loss = F.cross_entropy(image_logits, batch)
        text_loss = F.cross_entropy(text_logits, batch)
        image_text_loss = F.cross_entropy(image_text_logits, batch)

        # 加权因子 (KL值越大，权重越大)
        weight_factor = torch.exp(kl)  # 通过指数放大KL散度大的样本的影响

        # 最终的加权双模态损失
        weighted_image_text_loss = torch.mean((weight_factor)* image_text_loss)
        # 总损失：单模态损失 + 加权后的双模态损失
        total_loss = 0.3*image_loss + 0.6*text_loss + weighted_image_text_loss

        return total_loss





class OurClassfierConvnextV5(nn.Module):
    """
    这个版本分别使用 image 和 text 的 embedding 作为 kqv 然后再将输出进行融合。
    """
    def __init__(self, args, num_labels=2, loss_class="textimage_loss"):

        super(OurClassfierConvnextV5, self).__init__()
        self.text_encoder = BertEncoder()
        image_encoder_model = ConvNextForImageClassification.from_pretrained('./convnext-base-224')
        self.image_encoder = image_encoder_model.convnext
        self.conv = nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=1)
        self.textbased_cross_attention = CrossAttention_v2(dim=768)
        self.imagbased_cross_attention = CrossAttention_v2(dim=768)
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
        image_embedding = self.image_encoder(batch_data['transformed_image']).last_hidden_state
        image_embedding_reduced = self.conv(image_embedding)
        text_embedding_expanded = text_embedding.unsqueeze(dim=1)

        image_embedding_reduced = self.I2Iattention(image_embedding_reduced)
        image_embedding_pooled = self.avg_pool(image_embedding_reduced).view(batch_data['transformed_image'].shape[0], 768).unsqueeze(dim=1)

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
    dim = 64
    num_heads = 8
    model = MultiHeadCrossAttention_v2(dim, num_heads)

    x = torch.randn(32, 10, dim)  # (batch_size, seq_len_x, dim)
    y = torch.randn(32, 20, dim)  # (batch_size, seq_len_y, dim)

    output = model(x, y)
    print(output.shape)  # 输出形状: (32, 10, dim)
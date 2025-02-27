from transformers import ConvNextForImageClassification
import torch

model = ConvNextForImageClassification.from_pretrained('/mnt/sda1/zzixuantang/classfier_convNext/convnext-base-224')
model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=6)


if __name__ == "__main__":
    image = torch.zeros((1,3,224,224))
    print(model(image).logits.shape)
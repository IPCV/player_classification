import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torchvision import transforms


class TransformsWrapper:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']


def player_transform(type_='test', strength=1.0):
    if type_ == 'backbone':
        data_transform = A.Compose([A.RandomRotate90(p=0.05),
                                    A.RandomResizedCrop(224, 224, scale=(0.7, 1.)),
                                    A.ColorJitter(brightness=0.8 * strength,
                                                  contrast=0.8 * strength,
                                                  saturation=0.8 * strength,
                                                  hue=0.2 * strength,
                                                  always_apply=False,
                                                  p=0.8),
                                    A.ToGray(p=0.2),
                                    A.GaussianBlur(blur_limit=(23, 23), sigma_limit=(0.1, 2.), p=0.5),
                                    A.HorizontalFlip(p=0.5),
                                    A.ToFloat(255),
                                    ToTensorV2()
                                    ])
        data_transform = TransformsWrapper(data_transform)
    elif type_ == 'transformer':
        data_transform = A.Compose([A.RandomResizedCrop(224, 224, scale=(0.7, 1.)),
                                    A.HorizontalFlip(p=0.5),
                                    A.ToFloat(255),
                                    ToTensorV2()
                                    ])
        data_transform = TransformsWrapper(data_transform)
    else:
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    return data_transform


class RandomCoordsHorizontalFlip(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, coords):
        if torch.rand(1) < self.p:
            coords[:, 0] *= -1
        return coords

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

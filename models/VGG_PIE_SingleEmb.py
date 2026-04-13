

"""
In Paper PiRO (CVPR 2024), author used Pose-invariant Attention Network (PAN)
both in single and dual embedding model to train the model for effective output

In OWSC (CVPR 2026) paper, author used different approach,
In single embedding model, he used code of PIEs (CVPR 2018) and in
dual embedding model, he used PAN architecture

"""


"""
For single embedding model, in PIEs (CVPR 2018), 3 methods has been given 
1. PI-CNN
2. PI-Proxy
3. PI-Triplet Center


PI-Triplet Center = Triplet Center + Multi-view triplet center
                    Triplet Center = Triplet Loss + Center Loss

PI-Proxy = Proxy + Multi-view proxy
PI-CNN = CNN + Multi-view CNN
                    
"""

from torchvision.models.vgg import model_urls as model_url_vgg
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models

"""
PI-TC model with VGG backbone
"""


class VGG_avg_pitc(nn.Module):
    def __init__(self, input_view, output_class):
        super(VGG_avg_pitc, self).__init__()
        self.input_view = input_view
        VGG = models.vgg16(pretrained=True)
        VGG.classifier._modules['6'] = nn.Linear(
            4096, out_features=output_class)

        self.features = VGG.features
        self.input_view = input_view
        self.output_class = output_class
        self.classifier1 = nn.Sequential(*list(VGG.classifier)[0:5])
        self.class_centers = nn.Parameter(
            torch.randn(1, 1, self.output_class, 4096))

    def forward(self, x):
        image_features = None
        shape_feature = None
        _, views, _, _, _ = x.shape
        for view in range(views):
            view_feature = self.features(x[:, view]).view(x.shape[0], 25088)
            view_feature = self.classifier1(
                view_feature).view(x.shape[0], 1, 4096)

            if image_features is None:
                image_features = view_feature
            else:
                image_features = torch.cat([image_features, view_feature], 1)

        class_feature = F.normalize(self.class_centers, p=2, dim=3, eps=1e-12)
        class_feature = class_feature.view(self.output_class, 4096)

        shape_feature = torch.mean(image_features, 1)
        shape_feature = F.normalize(shape_feature, p=2, dim=1, eps=1e-12)

        image_features = image_features.view(x.shape[0]*views, -1)
        image_features = F.normalize(image_features, p=2, dim=1, eps=1e-12)

        return image_features, shape_feature, class_feature



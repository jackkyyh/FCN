import torch
from torch import nn
from torch.nn.init import kaiming_normal_
from torchvision import models
import torch.nn.functional as F
import numpy as np

class FCN8s_voc(nn.Module):
    def __init__(self, config):
        super(FCN8s_voc, self).__init__()
        """
        Referenced the implementation below
        https://github.com/zijundeng/pytorch-semantic-segmentation/
        """
        pretrained = True if config.mode == 'finetuning' else False
        self.num_class = config.num_class

        self.feature3, self.feature4, self.feature5, self.classifier = self._parse_vgg()
        self.score_pool3 = nn.Conv2d(256, self.num_class, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, self.num_class, kernel_size=1)
        self.score_fr = nn.Sequential(
             nn.Conv2d(512, 4096, kernel_size=7),
             nn.ReLU(inplace=True),
             nn.Dropout(),
             nn.Conv2d(4096, 4096, kernel_size=1),
             nn.ReLU(inplace=True),
             nn.Dropout(),
             nn.Conv2d(4096, self.num_class, kernel_size=1)
        )

        self.upscore2 = nn.ConvTranspose2d(self.num_class, self.num_class, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(self.num_class, self.num_class, kernel_size=4, stride=2, bias=False)
        #self.upscore8 = nn.UpsamplingBilinear2d(scale_factor = 8)
        self._init_weights()

    def _parse_vgg(self):
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        cnt = 0
        layers = []

        for layer in list(vgg.features):
            layers.append(layer)
            if isinstance(layer, nn.ReLU):
                layers[-1].inplace = True
            if isinstance(layer, nn.MaxPool2d):
                layers[-1].ceil_mode = True
                cnt+=1
                if cnt == 3:
                    layers[0].padding = (100,100)
                    pool3 = nn.Sequential(*layers)
                    layers = []
                elif cnt == 4:
                    pool4 = nn.Sequential(*layers)
                    layers = []
                elif cnt == 5:
                    pool5 = nn.Sequential(*layers)
                else:
                    continue
        return pool3, pool4, pool5, vgg.classifier

    def _get_upsampling_weight(self,in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        center = factor - 1 if  kernel_size % 2 == 1 else factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
        weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
        return torch.from_numpy(weight).float()

    def _init_weights(self):
        # score_fr: inject weights from classifier
        fc6_weight = self.classifier[0].weight.data.view(4096,512,7,7)
        fc6_bias = self.classifier[0].bias
        fc7_weight = self.classifier[3].weight.data.view(4096,4096,1,1)
        fc7_bias = self.classifier[3].bias
        self.score_fr[0].weight.data.copy_(fc6_weight)
        self.score_fr[0].bias.data.copy_(fc6_bias)
        self.score_fr[3].weight.data.copy_(fc7_weight)
        self.score_fr[3].bias.data.copy_(fc7_bias)
        kaiming_normal_(self.score_fr[6].weight.data, mode='fan_out', nonlinearity='relu')
        self.score_fr[6].bias.data.fill_(0)
        self.classifier = None

        self.upscore2.weight.data.copy_(self._get_upsampling_weight(self.num_class, self.num_class, 4))
        self.upscore_pool4.weight.data.copy_(self._get_upsampling_weight(self.num_class, self.num_class, 4))


    def forward(self, x):
        input_size = x.size()
        pool3 = self.feature3(x)
        pool4 = self.feature4(pool3)
        pool5 = self.feature5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(0.01 * pool4)
        score_pool3 = self.score_pool3(0.0001 * pool3)

        upscore_pool4 = self.upscore_pool4(score_pool4[:, :, 5:5+upscore2.size()[2], 5:5+upscore2.size()[3]]
                                          +upscore2)
        fuse = score_pool3[:, :, 9:9+upscore_pool4.size()[2], 9:9+upscore_pool4.size()[3]] + upscore_pool4
        out = F.interpolate(upscore_pool4,scale_factor = 8, mode='bilinear', align_corners=False)[:, :, 31:31+input_size[2], 31:31+input_size[3]].contiguous()#upscore8[:, :, 31:31+input_size[2], 31:31+input_size[3]].contiguous()
        return out
import torch
import torch.nn as nn

from slowfast.config.defaults import load_config
from slowfast.models.video_model_builder import ResNet


class Pooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        return self.pool(x[0]).view(x[0].size(0), -1)


def get_resnet():
    a = ResNet(load_config('./SlowFast/configs/contrastive_ssl/BYOL_SlowR50_16x4.yaml'))
    a.head = nn.Identity()
    s = torch.load('./initialization/BYOL_SlowR50_16x4_T4_epoch_00200.pyth')
    a.load_state_dict({k[len('backbone_hist.'):]: v for k, v in s['model_state'].items() if
                       k.startswith('backbone_hist.') and not k.startswith('backbone_hist.ssl_128')})
    a.head = Pooler()
    # with torch.no_grad():
    #     o = a([torch.randn(2, 3, 16, 224, 224)])  # => 2, 2048
    return a.eval()

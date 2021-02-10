"""
Grid-feat-vqa
"""
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.layers import FrozenBatchNorm2d
import torch
from torch import nn
from src.modeling.grid_feats import add_attribute_config
import os
import horovod.torch as hvd


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1,
            init_identity=False):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)
    if init_identity:
        """ init as identity kernel, not working yet
        0 0 0
        0 1 0
        0 0 0
        """
        identity_weight = conv.weight.new_zeros(3, 3)
        identity_weight[1, 1] = 1. / in_planes
        identity_weight = identity_weight.view(
            1, 1, 3, 3).expand(conv.weight.size())
        with torch.no_grad():
            conv.weight = nn.Parameter(identity_weight)
    return conv


class GridFeatBackbone(nn.Module):
    def __init__(self, detectron2_model_cfg, config,
                 input_format="BGR"):
        super(GridFeatBackbone, self).__init__()
        self.detectron2_cfg = self.__setup__(detectron2_model_cfg)
        self.feature = build_model(self.detectron2_cfg)
        self.grid_encoder = nn.Sequential(
            conv3x3(config.backbone_channel_in_size,
                    config.hidden_size),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.input_format = input_format
        assert input_format == "BGR", "detectron 2 image input format should be BGR"
        
        self.config = config

    def __setup__(self, config_file):
        """
        Create configs and perform basic setups.
        """
        rank = hvd.rank()
        detectron2_cfg = get_cfg()
        add_attribute_config(detectron2_cfg)
        detectron2_cfg.merge_from_file(config_file)
        # force the final residual block to have dilations 1
        detectron2_cfg.MODEL.RESNETS.RES5_DILATION = 1
        # FIXME: hardcode device to each rank
        detectron2_cfg.MODEL.DEVICE = f'cpu'
        detectron2_cfg.freeze()

        setup_logger(None, distributed_rank=rank, name="fvcore")
        logger = setup_logger(None, distributed_rank=rank)
        return detectron2_cfg

    def load_state_dict(self, model_path):
        if not os.path.exists(model_path):
            print(f"{model_path} does not exist, loading ckpt from detectron2")
            DetectionCheckpointer(
                self.feature).resume_or_load(
                    self.detectron2_cfg.MODEL.WEIGHTS, resume=True)
        else:
            DetectionCheckpointer(self.feature).resume_or_load(
                    model_path, resume=True)

    @property
    def config_file(self):
        return self.detectron2_cfg.dump()

    def train(self, mode=True):
        super(GridFeatBackbone, self).train(mode)

    def forward(self, x):
        bsz, n_frms, c, h, w = x.shape
        x = x.view(bsz*n_frms, c, h, w)
        if self.input_format == "BGR":
            # RGB->BGR, images are read in as RGB by default
            x = x[:, [2, 1, 0], :, :]
        res5_features = self.feature.backbone(x)
        grid_feat_outputs = self.feature.roi_heads.get_conv5_features(
            res5_features)

        grid = self.grid_encoder(grid_feat_outputs)  # (B * n_frm, C, H, W)
        new_c, new_h, new_w = grid.shape[-3:]
        # if n_frms != 0:
        grid = grid.view(bsz, n_frms, new_c, new_h, new_w)  # (B, n_frm, C, H, W)

        grid = grid.permute(0, 1, 3, 4, 2)  # (B, n_frm=3, H, W, C)
        return grid

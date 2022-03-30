from torch import nn as nn

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmdet3d.models.detectors.base import Base3DDetector


@DETECTORS.register_module()
class SingleStage3DDetector_cl(Base3DDetector):
    """SingleStage3DDetector.

    This class serves as a base class for single-stage 3D detectors.

    Args:
        backbone (dict): Config dict of detector's backbone.
        neck (dict, optional): Config dict of neck. Defaults to None.
        bbox_head (dict, optional): Config dict of box head. Defaults to None.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        pretrained (str, optional): Path of pretrained models.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 img_backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 pretrained_img=None,
                 pretraining=None):
        super(SingleStage3DDetector_cl, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        self.pretraining = pretraining
        if not self.pretraining:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)

        else:
            self.img_backbone = build_backbone(img_backbone)
            self.init_weights_img(pretrained_img=pretrained_img)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
        

    # used in model train after pretraining
    def init_weights(self, pretrained=None):
        """Initialize weights of detector."""
        super(SingleStage3DDetector_cl, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        
        if not self.pretraining:
            self.bbox_head.init_weights()

    # TODO: may be used in feature ablative analysis 
    def init_weights_img(self, pretrained_img=None):
        """Initialize weights of detector."""
        super(SingleStage3DDetector_cl, self).init_weights(pretrained_img)
        self.img_backbone.init_weights(pretrained=pretrained_img)


    def extract_feat(self, points, img_metas=None, img=None):
        """Directly extract features from the backbone+neck.

        Args:
            points (torch.Tensor): Input points.
        """
        if not self.pretraining:
            x = self.backbone(points)
            if self.with_neck:
                x = self.neck(x)
            return x
        else:
            x = self.backbone(points)
            img_x = self.img_backbone(img)
            return x, img_x

    def extract_feats(self, points, img_metas):
        """Extract features of multiple samples."""
        return [
            self.extract_feat(pts, img_meta)
            for pts, img_meta in zip(points, img_metas)
        ]

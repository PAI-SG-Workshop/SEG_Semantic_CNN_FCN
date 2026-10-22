"""
Semantic segmentation models (semseg task), all trained from scratch under the common protocol.

FCN-ResNet50        Long, Shelhamer & Darrell, "Fully Convolutional Networks for Semantic Segmentation" (CVPR 2015);
                    torchvision 0.20.1 segmentation.fcn_resnet50 (dilated ResNet-50, output stride 8, FCNHead
                    conv3x3 512 + BN + ReLU + dropout 0.1 + 1x1; aux FCNHead on layer3, weight 0.5 as in the
                    torchvision reference recipe) - torchvision is the approved source (original is Caffe).

[Unified] no ImageNet weights (the papers all start from ImageNet), common init, no mean / std, 608 input with the
YOLO-paper augmentation, Adam 1e-4 constant, cross-entropy with ignore 255 on the logits upsampled to the input,
PreciseBN for BN models (micro batch 2), K = C + 1 classes.
"""


from torchvision.models.segmentation import fcn_resnet50

from detectors.semseg.common import SemSegBase


class _PreciseBN:
    precise_bn, precise_bn_images, precise_bn_batch_size = True, 1600, 16


class TorchvisionSeg(_PreciseBN, SemSegBase):
    aux_weight = 0.5

    def __init__(self, num_classes, ctor):
        super().__init__(num_classes)
        self.net = ctor(weights=None, weights_backbone=None, num_classes=self.K, aux_loss=True)

    def logits(self, x):
        out = self.net(x)
        return out["out"], out.get("aux") if self.training else None


class FCN(TorchvisionSeg):
    def __init__(self, num_classes):
        super().__init__(num_classes, fcn_resnet50)


# ---------------------------------------------------------------- SegFormer (NVlabs mix_transformer.py)


MODELS = {"fcn": FCN}   # [Release] this repository's model only


def make_build(name):
    def build_model(task, num_classes):
        return MODELS[name](num_classes)
    return build_model

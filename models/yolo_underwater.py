import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_layers import conv1x1, conv3x3, EP, PEP, FCA, YOLOLayer
from .ghost_module import GhostModule, GhostBottleneck, CheapOps, Mix
from .preprocessing_module import Preprocessing


class YOLO_Underwater(nn.Module):
    def __init__(self, num_classes, image_size, use_preprocessing=False):
        super(YOLO_Underwater, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_anchors = 3
        self.yolo_channels = (self.num_classes + 5) * self.num_anchors

        anchors52 = [[17, 24], [24, 37], [28, 52]]
        anchors26 = [[40, 45], [39, 66], [55, 61]]
        anchors13 = [[50, 89], [71, 111], [120, 167]]

        self.use_preprocessing = use_preprocessing
        if self.use_preprocessing:
            self.preprocessing = Preprocessing(input_channels=3)

        self.conv1 = conv3x3(3, 16, stride=1)
        self.ep1 = EP(16, 16)  # contoh EP modul
        self.conv2 = GhostModule(16, 32, ratio=2, kernel_size=3, stride=2)
        self.pep1 = PEP(32, 32, 32, se_ratio=0.25, ghost_ratio=2)
        self.ep2 = EP(32, 64, stride=2)
        self.pep3 = PEP(64, 96, 64, se_ratio=0.25, ghost_ratio=2)
        self.ep3 = EP(64, 128, stride=2)
        self.pep6 = PEP(128, 192, 128, se_ratio=0.25, ghost_ratio=2)
        self.fca1 = FCA(192, reduction_ratio=16)
        self.ep4 = EP(192, 256, stride=2)
        self.pep9 = PEP(256, 512, 256, se_ratio=0.25, ghost_ratio=2)
        self.fca2 = FCA(512, reduction_ratio=16)
        self.pep12 = PEP(512, 768, 512, se_ratio=0)
        self.chp_op1 = CheapOps(768, 128)
        self.mix1 = Mix(inp=896, oup=128)  # 768 + 128
        self.ep5 = EP(128, 128)
        self.pep17 = PEP(128, 1024, 128, se_ratio=0.25, ghost_ratio=2)
        self.conv5 = GhostBottleneck(128, 1024, 128, se_ratio=0, ghost_ratio=2, stride=1)
        self.chp_op2 = CheapOps(1024, 128)
        self.mix2 = Mix(inp=1152, oup=128)  # 1024 + 128
        self.conv6 = conv1x1(256, 128, stride=1)
        self.pep19 = PEP(384, 768, 256)
        self.fca3 = FCA(768, reduction_ratio=16)
        self.conv7 = conv1x1(256, 128, stride=1)
        self.conv8 = conv1x1(128, 64, stride=1)
        self.pep20 = PEP(192, 512, 128)
        self.fca4 = FCA(512, reduction_ratio=16)
        self.pep22_reg_iou = PEP(128, 512, 128, se_ratio=0.25, ghost_ratio=2)
        self.pep22_cls = PEP(128, 512, 128, se_ratio=0.25, ghost_ratio=2)
        self.conv9_reg = conv1x1(128, 4 * self.num_anchors, stride=1, bn=False)
        self.conv9_iou = conv1x1(128, self.num_anchors, stride=1, bn=False)
        self.conv9_cls = conv1x1(128, self.num_classes * self.num_anchors, stride=1, bn=False)
        self.yolo_layer52 = YOLOLayer(anchors52, num_classes, img_dim=image_size)
        self.ep6_reg_iou = EP(128, 768, stride=1)
        self.ep6_cls = EP(128, 768, stride=1)
        self.conv10_reg = conv1x1(256, 4 * self.num_anchors, stride=1, bn=False)
        self.conv10_iou = conv1x1(256, self.num_anchors, stride=1, bn=False)
        self.conv10_cls = conv1x1(256, self.num_classes * self.num_anchors, stride=1, bn=False)
        self.yolo_layer26 = YOLOLayer(anchors26, num_classes, img_dim=image_size)
        self.ep7_reg_iou = EP(256, 1024, stride=1)
        self.ep7_cls = EP(256, 1024, stride=1)
        self.conv11_reg = conv1x1(512, 4 * self.num_anchors, stride=1, bn=False)
        self.conv11_iou = conv1x1(512, self.num_anchors, stride=1, bn=False)
        self.conv11_cls = conv1x1(512, self.num_classes * self.num_anchors, stride=1, bn=False)
        self.yolo_layer13 = YOLOLayer(anchors13, num_classes, img_dim=image_size)
        self.yolo_layers = [self.yolo_layer52, self.yolo_layer26, self.yolo_layer13]

    def forward(self, x, targets=None, indexes=None):
        loss = 0
        yolo_outputs = []
        image_size = x.size(2)

        if self.use_preprocessing:
            x = self.preprocessing(x)

        out = self.conv1(x)
        out = self.ep1(out)
        out = self.conv2(out)
        out = self.pep1(out)
        out = self.ep2(out)
        out = self.pep3(out)
        out = self.ep3(out)
        out = self.pep6(out)
        out_pep6 = out  # Simpan output dari pep6 untuk nanti di concat

        out = self.fca1(out)
        out = self.ep4(out)
        out = self.pep9(out)
        out = self.fca2(out)
        out = self.pep12(out)
        chp_op1 = self.chp_op1(out)
        mix1 = self.mix1(blocks=[out, chp_op1], target=chp_op1)
        cat_1 = torch.cat([out, mix1], dim=1)

        out = self.ep5(cat_1)
        out = self.pep17(out)
        out_conv5 = self.conv5(out)
        chp_op2 = self.chp_op2(out)
        mix2 = self.mix2(blocks=[out_conv5, chp_op2], target=chp_op2)
        cat_2 = torch.cat([out_conv5, mix2], dim=1)

        out = F.interpolate(self.conv6(cat_2), scale_factor=2)
        out = torch.cat([out, cat_1], dim=1)
        out = self.pep19(out)
        out = self.fca3(out)

        out_conv7 = self.conv7(out)
        out = F.interpolate(self.conv8(out_conv7), scale_factor=2)
        out = torch.cat([out, out_pep6], dim=1)  # Menggunakan out_pep6 yang sudah disimpan
        out = self.pep20(out)
        out = self.fca4(out)

        out_reg_iou = self.pep22_reg_iou(out)
        out_cls = self.pep22_cls(out)
        out_conv9_reg = self.conv9_reg(out_reg_iou)
        out_conv9_iou = self.conv9_iou(out_reg_iou)
        out_conv9_cls = self.conv9_cls(out_cls)
        out_conv9 = torch.cat([out_conv9_reg, out_conv9_iou, out_conv9_cls], dim=1)
        temp, layer_loss = self.yolo_layer52(out_conv9, targets, indexes, image_size)
        loss += layer_loss
        yolo_outputs.append(temp)

        out_reg_iou = self.ep6_reg_iou(out_conv7)
        out_cls = self.ep6_cls(out_conv7)
        out_conv10_reg = self.conv10_reg(out_reg_iou)
        out_conv10_iou = self.conv10_iou(out_reg_iou)
        out_conv10_cls = self.conv10_cls(out_cls)
        out_conv10 = torch.cat([out_conv10_reg, out_conv10_iou, out_conv10_cls], dim=1)
        temp, layer_loss = self.yolo_layer26(out_conv10, targets, indexes, image_size)
        loss += layer_loss
        yolo_outputs.append(temp)

        out_reg_iou = self.ep7_reg_iou(cat_2)
        out_cls = self.ep7_cls(cat_2)
        out_conv11_reg = self.conv11_reg(out_reg_iou)
        out_conv11_iou = self.conv11_iou(out_reg_iou)
        out_conv11_cls = self.conv11_cls(out_cls)
        out_conv11 = torch.cat([out_conv11_reg, out_conv11_iou, out_conv11_cls], dim=1)
        temp, layer_loss = self.yolo_layer13(out_conv11, targets, indexes, image_size)
        loss += layer_loss
        yolo_outputs.append(temp)

        yolo_outputs = torch.cat(yolo_outputs, 1)
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def name(self):
        return "YOLO-Underwater"

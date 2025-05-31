import torch
import torch.nn as nn
from utils.stats import build_targets, to_cpu

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        probas = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probas, 1 - probas)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = FocalLoss(alpha=1.0, gamma=2.0)
        self.obj_scale = 1
        self.noobj_scale = 10
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / grid_size
        g = grid_size
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(w / self.stride, h / self.stride) for w, h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, indexes=None, img_dim=None):
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Decode components
        x_center = torch.sigmoid(prediction[..., 0])
        y_center = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        raw_conf = prediction[..., 4]          # logit
        raw_cls = prediction[..., 5:]          # logit

        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x_center.data + self.grid_x
        pred_boxes[..., 1] = y_center.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat([
            pred_boxes.view(num_samples, -1, 4) * self.stride,
            torch.sigmoid(raw_conf).view(num_samples, -1, 1),
            torch.sigmoid(raw_cls).view(num_samples, -1, self.num_classes)
        ], -1)

        if targets is None:
            return output, 0

        # Training mode
        targets = targets.squeeze(dim=0)
        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
            pred_boxes=pred_boxes,
            pred_cls=torch.sigmoid(raw_cls),
            target=targets,
            index=indexes,
            anchors=self.scaled_anchors,
            ignore_thres=self.ignore_thres,
        )

        # Clamp logit to avoid extreme gradient explosions
        raw_conf = torch.clamp(raw_conf, -10, 10)
        raw_cls = torch.clamp(raw_cls, -10, 10)

        loss_x = self.mse_loss(x_center[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y_center[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

        loss_conf_obj = self.bce_loss(raw_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(raw_conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj

        loss_cls = self.bce_loss(raw_cls[obj_mask], tcls[obj_mask])

        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        # Metrics (using sigmoid to get probas)
        with torch.no_grad():
            pred_conf = torch.sigmoid(raw_conf)
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        self.metrics = {
            "loss": to_cpu(total_loss).item(),
            "x": to_cpu(loss_x).item(),
            "y": to_cpu(loss_y).item(),
            "w": to_cpu(loss_w).item(),
            "h": to_cpu(loss_h).item(),
            "conf": to_cpu(loss_conf).item(),
            "cls": to_cpu(loss_cls).item(),
            "cls_acc": to_cpu(class_mask[obj_mask].float().mean()).item(),
            "recall50": to_cpu(recall50).item(),
            "recall75": to_cpu(recall75).item(),
            "precision": to_cpu(precision).item(),
            "conf_obj": to_cpu(conf_obj).item(),
            "conf_noobj": to_cpu(conf_noobj).item(),
            "grid_size": grid_size,
        }

        return output, total_loss

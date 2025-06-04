import os
import numpy as np
import torch
from terminaltables import AsciiTable
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.stats import (
    non_max_suppression, xywh2xyxy,
    get_batch_statistics, ap_per_class, load_classe_names
)

@torch.no_grad()
def val(model, optimizer, scheduler, dataloader, epoch, opt, val_logger, best_mAP=0):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() and opt.gpu else 'cpu')
    ngpu = torch.cuda.device_count() if device.type == 'cuda' else 1

    labels = []
    sample_matrics = []
    total_loss = []

    coco = COCO(os.path.join(opt.dataset_path, f"val_fixed.json"))
    coco_dt = []

    for i, (images, targets, indexes) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        targets = targets.to(device)
        indexes = indexes.to(device)

        rep_targets = []
        for _ in range(ngpu):
            rep_targets.append(targets.unsqueeze(dim=0))
        rep_targets = torch.cat(rep_targets, dim=0).to(device)

        loss, detections = model(images, rep_targets, indexes)

        # Skip this batch if detections are empty
        if detections is None or len(detections) == 0:
            continue

        detections = non_max_suppression(detections, opt.conf_thresh, opt.nms_thresh)

        # Skip if no valid detections after NMS
        if detections is None or len(detections) == 0:
            continue

        # Populate sample_matrics only if detections are valid
        sample_matrics += get_batch_statistics(detections, targets, indexes, iou_threshold=0.5)

    # Ensure sample_matrics has enough data before unpacking
    if len(sample_matrics) > 0:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_matrics))]
    else:
        print("Warning: No valid detections to compute metrics.")
        return best_mAP  # Skip evaluation if no detections

    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # COCOeval
    coco_results = coco.loadRes(coco_dt)
    coco_eval = COCOeval(coco, coco_results, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Print the results in the same format as COCOeval
    metrics = {
        "precision": precision.mean(),
        "recall": recall.mean(),
        "f1": f1.mean(),
        "mAP": AP.mean(),
        "loss": np.array(total_loss).mean(),
    }

    # mAP50, mAP75, mAPsmall, mAPmedium, mAPlarge
    mAP50 = coco_eval.stats[0]  # mAP at IoU=0.50
    mAP75 = coco_eval.stats[1]  # mAP at IoU=0.75
    mAPsmall = coco_eval.stats[3]  # mAP at small objects
    mAPmedium = coco_eval.stats[4]  # mAP at medium objects
    mAPlarge = coco_eval.stats[5]  # mAP at large objects

    print(f"mAP50: {mAP50:.3f}, mAP75: {mAP75:.3f}, mAPsmall: {mAPsmall:.3f}, mAPmedium: {mAPmedium:.3f}, mAPlarge: {mAPlarge:.3f}")

    metric_table_data = [
        ['Metrics', 'Value'],
        ['precision', precision.mean()],
        ['recall', recall.mean()],
        ['f1', f1.mean()],
        ['mAP', AP.mean()],
        ['mAP50', mAP50],
        ['mAP75', mAP75],
        ['mAPsmall', mAPsmall],
        ['mAPmedium', mAPmedium],
        ['mAPlarge', mAPlarge],
        ['loss', np.array(total_loss).mean()]
    ]

    # Print detailed class-wise AP
    class_names = load_classe_names(opt.classname_path)
    for i, c in enumerate(ap_class):
        metric_table_data += [['AP-{}'.format(class_names[c]), AP[i]]]
    
    metric_table = AsciiTable(
        metric_table_data,
        title='[Epoch {:d}/{:d}]'.format(epoch, opt.num_epochs)
    )

    val_logger.print_and_write(f'{metric_table.table}\n')

    if best_mAP < AP.mean():
        save_file_path = os.path.join(opt.checkpoint_path, 'best.pt')  # Save as .pt file
        states = {
            'epoch': epoch + 1,
            'model': opt.model,
            'state_dict': model.module.state_dict() if ngpu > 1 else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_mAP': best_mAP,
        }
        torch.save(states, save_file_path)  # Save best model as .pt
        best_mAP = AP.mean()

    print("current best mAP:" + str(best_mAP))

    return best_mAP

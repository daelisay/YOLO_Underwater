import math
import time
import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def to_cpu(tensor):
	return tensor.detach().cpu()


def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm2d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)


def xywh2xyxy(x):
	y = x.new(x.shape)
	y[..., 0] = x[..., 0] - x[..., 2] / 2
	y[..., 1] = x[..., 1] - x[..., 3] / 2
	y[..., 2] = x[..., 0] + x[..., 2] / 2
	y[..., 3] = x[..., 1] + x[..., 3] / 2
	return y


def load_classe_names(classname_path):
	class_names = []
	with open(classname_path, "r") as fp:
		lines = fp.readlines()
	for line in lines:
		line = line.strip()
		if line:
			class_names.append(line)
	return class_names


def ap_per_class(tp, conf, pred_cls, target_cls):
	""" Compute the average precision, given the recall and precision curves.
	Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
	# Arguments
		tp:	True positives (list).
		conf:  Objectness value from 0-1 (list).
		pred_cls: Predicted object classes (list).
		target_cls: True object classes (list).
	# Returns
		The average precision as computed in py-faster-rcnn.
	"""

	# Sort by objectness
	i = np.argsort(-conf)
	tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

	# Find unique classes
	unique_classes = np.unique(target_cls)

	# Create Precision-Recall curve and compute AP for each class
	ap, p, r = [], [], []
	for c in unique_classes:
		i = pred_cls == c
		n_gt = (target_cls == c).sum()  # Number of ground truth objects
		n_p = i.sum()  # Number of predicted objects

		# print(n_p, n_gt)

		if n_p == 0 and n_gt == 0:
			continue
		elif n_p == 0 or n_gt == 0:
			ap.append(0)
			r.append(0)
			p.append(0)
		else:
			# Accumulate FPs and TPs
			fpc = (1 - tp[i]).cumsum()
			tpc = (tp[i]).cumsum()

			# Recall
			recall_curve = tpc / (n_gt + 1e-16)
			r.append(recall_curve[-1])

			# Precision
			precision_curve = tpc / (tpc + fpc)
			p.append(precision_curve[-1])

			# AP from recall-precision curve
			ap.append(compute_ap(recall_curve, precision_curve))

	# Compute F1 score (harmonic mean of precision and recall)
	p, r, ap = np.array(p), np.array(r), np.array(ap)
	f1 = 2 * p * r / (p + r + 1e-16)

	return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
	""" Compute the average precision, given the recall and precision curves.
	Code originally from https://github.com/rbgirshick/py-faster-rcnn.
	# Arguments
		recall:	The recall curve (list).
		precision: The precision curve (list).
	# Returns
		The average precision as computed in py-faster-rcnn.
	"""
	# correct AP calculation
	# first append sentinel values at the end
	mrec = np.concatenate(([0.0], recall, [1.0]))
	mpre = np.concatenate(([0.0], precision, [0.0]))

	# compute the precision envelope
	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

	# to calculate area under PR curve, look for points
	# where X axis (recall) changes value
	i = np.where(mrec[1:] != mrec[:-1])[0]

	# and sum (\Delta recall) * prec
	ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
	return ap


def get_batch_statistics(outputs, targets, indexes, iou_threshold):
	targets = to_cpu(targets)
	""" Compute true positives, predicted scores and predicted labels per sample """
	batch_metrics = []
	# print(indexes.shape, indexes)
	index_dict = {}
	for i in range(indexes.size(0)):
		index_dict[i] = indexes[i].item()
	# print('get_batch_statistics', indexes.device, index_dict)

	# unlist = []
	# for i in targets[:, 0]:
	# 	if i.item() not in index_dict.keys():
	# 		index_dict[int(i.item())] = -1
	# 		unlist.append(int(i.item()))
	# print('get_batch_stat unlist:',indexes.device, len(unlist), unlist)

	# print('output', len(outputs), outputs)
	for sample_i in range(len(outputs)):

		if outputs[sample_i] is None:
			continue

		output = outputs[sample_i]
		# print(output[0, :4], output[0, 4], output[0, -1])
		pred_boxes = to_cpu(output[:, :4])
		pred_scores = to_cpu(output[:, 4])
		pred_labels = to_cpu(output[:, -1])

		true_positives = np.zeros(pred_boxes.shape[0])

		# annotations = []
		# for target in targets:
		# 	if index_dict[int(target[0].item())] == sample_i:
		# 		annotations.append(target)
		# annotations = torch.stack(annotations)[:, 1:]
		annotations = targets[targets[:, 0] == index_dict[sample_i]][:, 1:]
		# print(annotations)
		target_labels = annotations[:, 0] if len(annotations) else []
		if len(annotations):
			detected_boxes = []
			target_boxes = annotations[:, 1:]

			for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

				# If targets are found break
				if len(detected_boxes) == len(annotations):
					break

				# Ignore if label is not one of the target labels
				# print(pred_label, target_labels)
				if pred_label not in target_labels:
					continue

				iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
				if iou >= iou_threshold and box_index not in detected_boxes:
					true_positives[pred_i] = 1
					detected_boxes += [box_index]
		# print('tp ...: ', true_positives, pred_scores, pred_labels)
		batch_metrics.append([true_positives, pred_scores, pred_labels])
	if len(batch_metrics) == 0:
		batch_metrics = [[[], [], []]]
	return batch_metrics


def bbox_wh_iou(wh1, wh2):
	wh2 = wh2.t()
	w1, h1 = wh1[0], wh1[1]
	w2, h2 = wh2[0], wh2[1]
	inter_area = torch.min(w1, w2) * torch.min(h1, h2)
	union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
	return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
	"""
	Returns the IoU of two bounding boxes
	"""
	if not x1y1x2y2:
		# Transform from center and width to exact coordinates
		b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
		b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
		b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
		b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
	else:
		# Get the coordinates of bounding boxes
		b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
		b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

	# get the corrdinates of the intersection rectangle
	inter_rect_x1 = torch.max(b1_x1, b2_x1)
	inter_rect_y1 = torch.max(b1_y1, b2_y1)
	inter_rect_x2 = torch.min(b1_x2, b2_x2)
	inter_rect_y2 = torch.min(b1_y2, b2_y2)
	# Intersection area
	inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
		inter_rect_y2 - inter_rect_y1 + 1, min=0
	)
	# Union Area
	b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
	b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

	iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

	return iou


def non_max_suppression(prediction,
						conf_thres=0.25,
						iou_thres=0.45,
						classes=None,
						agnostic=False,
						multi_label=False,
						labels=(),
						max_det=300):
	"""Non-Maximum Suppression (NMS) on inference results to reject overlapping
	bounding boxes
	Returns:
		 list of detections, on (n,6) tensor per image [xyxy, conf, cls]
	"""

	bs = prediction.shape[0]  # batch size
	nc = prediction.shape[2] - 5  # number of classes
	xc = prediction[..., 4] > conf_thres  # candidates

	# Checks
	assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
	assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

	# Settings
	# min_wh = 2  # (pixels) minimum box width and height
	max_wh = 7680  # (pixels) maximum box width and height
	max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
	time_limit = 0.3 + 0.03 * bs  # seconds to quit after
	redundant = True  # require redundant detections
	multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
	merge = False  # use merge-NMS

	t = time.time()
	output = [torch.zeros((0, 6), device=prediction.device)] * bs
	for xi, x in enumerate(prediction):  # image index, image inference
		# Apply constraints
		# x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
		x = x[xc[xi]]  # confidence

		# Cat apriori labels if autolabelling
		if labels and len(labels[xi]):
			lb = labels[xi]
			v = torch.zeros((len(lb), nc + 5), device=x.device)
			v[:, :4] = lb[:, 1:5]  # box
			v[:, 4] = 1.0  # conf
			v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
			x = torch.cat((x, v), 0)

		# If none remain process next image
		if not x.shape[0]:
			continue

		# Compute conf
		x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

		# Box (center x, center y, width, height) to (x1, y1, x2, y2)
		box = xywh2xyxy(x[:, :4])

		# Detections matrix nx6 (xyxy, conf, cls)
		if multi_label:
			i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
			x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
		else:  # best class only
			conf, j = x[:, 5:].max(1, keepdim=True)
			x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

		# Filter by class
		if classes is not None:
			x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

		# Apply finite constraint
		# if not torch.isfinite(x).all():
		#     x = x[torch.isfinite(x).all(1)]

		# Check shape
		n = x.shape[0]  # number of boxes
		if not n:  # no boxes
			continue
		elif n > max_nms:  # excess boxes
			x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

		# Batched NMS
		c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
		boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
		i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
		if i.shape[0] > max_det:  # limit detections
			i = i[:max_det]
		if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
			# update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
			iou = bbox_iou(boxes[i], boxes) > iou_thres  # iou matrix
			weights = iou * scores[None]  # box weights
			x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
			if redundant:
				i = i[iou.sum(1) > 1]  # require redundancy

		output[xi] = x[i]
		if (time.time() - t) > time_limit:
			print(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
			break  # time limit exceeded

	return output


# def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
# 	"""
# 	Removes detections with lower object confidence score than 'conf_thres' and performs
# 	Non-Maximum Suppression to further filter detections.
# 	Returns detections with shape:
# 		(x1, y1, x2, y2, object_conf, class_score, class_pred)
# 	"""
#
# 	# From (center x, center y, width, height) to (x1, y1, x2, y2)
#
# 	prediction[..., :4] = xywh2xyxy(prediction[..., :4])
# 	output = [None for _ in range(len(prediction))]
# 	for image_i, image_pred in enumerate(prediction):
# 		# Filter out confidence scores below threshold
# 		# print(image_pred[:, 4])
# 		image_pred = image_pred[image_pred[:, 4] >= conf_thres]
# 		# If none are remaining => process next image
# 		if not image_pred.size(0):
# 			continue
# 		# Object confidence times class confidence
# 		score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
# 		# Sort by it
# 		image_pred = image_pred[(-score).argsort()]
# 		class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
# 		detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
# 		# Perform non-maximum suppression
# 		keep_boxes = []
# 		while detections.size(0):
# 			large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
# 			label_match = detections[0, -1] == detections[:, -1]
# 			# Indices of boxes with lower confidence scores, large IOUs and matching labels
# 			invalid = large_overlap & label_match
# 			weights = detections[invalid, 4:5]
# 			# Merge overlapping bboxes by order of confidence
# 			detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
# 			keep_boxes += [detections[0]]
# 			detections = detections[~invalid]
# 		if keep_boxes:
# 			output[image_i] = torch.stack(keep_boxes)
#
# 	return output


def build_targets(pred_boxes, pred_cls, target, index, anchors, ignore_thres):
	# print(pred_boxes.size(), pred_cls.size(), target.size(), anchors.size())
	pred_boxes = pred_boxes.clone()
	pred_cls = pred_cls.clone()
	target = target.clone()
	anchors = anchors.clone()

	# ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
	# FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

	ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
	FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

	nB = pred_boxes.size(0)
	assert nB == index.size(0)
	nA = pred_boxes.size(1)
	nC = pred_cls.size(-1)
	nG = pred_boxes.size(2)

	# print(nB, nA, nC, nG)

	# Output tensors
	obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
	noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
	class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
	iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
	tx = FloatTensor(nB, nA, nG, nG).fill_(0)
	ty = FloatTensor(nB, nA, nG, nG).fill_(0)
	tw = FloatTensor(nB, nA, nG, nG).fill_(0)
	th = FloatTensor(nB, nA, nG, nG).fill_(0)
	tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

	# Convert to position relative to box
	target_boxes = target[:, 2:6] * nG
	gxy = target_boxes[:, :2]
	gwh = target_boxes[:, 2:]
	# Get anchors with best iou
	ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
	# print(ious.size(), gwh.size(), target.size())

	best_ious, best_n = ious.max(0)
	# print(best_ious.size(), best_n.size())

	# Separate target values
	index_dic = {}
	for i in range(index.size(0)):
		index_dic[int(index[i].item())] = i
	# print('build',index.device, index_dic)

	b, target_labels = target[:, :2].long().t()

	un_list = []
	for b_ in b:
		if int(b_.item()) not in index_dic.keys():
			index_dic[int(b_.item())] = -1
			un_list.append(int(b_.item()))
	# print('build_target un_list',index.device,len(un_list), un_list)

	b = torch.tensor([index_dic[int(b_.item())] for b_ in b])
	gx, gy = gxy.t()
	gw, gh = gwh.t()
	gi, gj = gxy.long().t()
	# Set masks

	# print(b, best_n, gj, gi)
	obj_mask[b, best_n, gj, gi] = 1
	noobj_mask[b, best_n, gj, gi] = 0

	# Set noobj mask to zero where iou exceeds ignore threshold
	for i, anchor_ious in enumerate(ious.t()):
		# print(noobj_mask.size())
		# print(b[i], anchor_ious>ignore_thres, gj[i], gi[i])
		noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

	# Coordinates
	tx[b, best_n, gj, gi] = gx - gx.floor()
	ty[b, best_n, gj, gi] = gy - gy.floor()
	# Width and height
	tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
	th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

	# print(tw[b,best_n, gj, gi])

	# One-hot encoding of label
	tcls[b, best_n, gj, gi, target_labels] = 1
	# Compute label correctness and iou at best anchor
	class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
	iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

	tconf = obj_mask.float()

	# iou_scores = iou_scores.cuda()
	# class_mask = class_mask.cuda()
	# obj_mask = obj_mask.cuda()
	obj_mask = obj_mask.bool()
	# noobj_mask = noobj_mask.cuda()
	noobj_mask = noobj_mask.bool()
	# tx = tx.cuda()
	# ty = ty.cuda()
	# tw = tw.cuda()
	# th = th.cuda()
	# tcls = tcls.cuda()
	# tconf = tconf.cuda()

	return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
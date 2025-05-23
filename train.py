import os
import torch
from terminaltables import AsciiTable

from tqdm import tqdm


def train(model, optimizer, scheduler, dataloader, epoch, opt, logger, best_mAP=0):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() and opt.gpu else "cpu")
    ngpu = torch.cuda.device_count() if device.type == "cuda" else 1

    for i, (images, targets, indexes) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()

        # Pastikan targets 2D tensor
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)

        # Skip batch jika targets kosong
        if targets.numel() == 0:
            continue

        # Pindahkan tensor ke device
        images = images.to(device)
        targets = targets.to(device)
        indexes = indexes.to(device)

        rep_targets = []
        for _ in range(ngpu):
            rep_targets.append(targets.unsqueeze(0))
        rep_targets = torch.cat(rep_targets, dim=0).to(device)

        loss, detections = model(images, rep_targets, indexes)

        if ngpu > 1:
            loss = loss.sum()

        loss.backward()
        optimizer.step()

        # Logging
        if ngpu > 1:
            metric_keys = model.module.yolo_layers[0].metrics.keys()
            yolo_metrics = [model.module.yolo_layers[i].metrics for i in range(len(model.module.yolo_layers))]
        else:
            metric_keys = model.yolo_layers[0].metrics.keys()
            yolo_metrics = [model.yolo_layers[i].metrics for i in range(len(model.yolo_layers))]

        layer_header = ['YOLO Layer {}'.format(i) for i in range(len(yolo_metrics))]
        metric_table_data = [['Metrics', *layer_header]]
        formats = {m: '%.6f' for m in metric_keys}
        for metric in metric_keys:
            row_metrics = [formats[metric] % ym.get(metric, 0) for ym in yolo_metrics]
            metric_table_data += [[metric, *row_metrics]]
        metric_table_data += [['total loss', '{:.6f}'.format(loss.item()), '', '']]

        metric_table = AsciiTable(
            metric_table_data,
            title='[Epoch {:d}/{:d}, Batch {:d}/{:d}, Current best mAP {:.4f}]'.format(
                epoch, opt.num_epochs, i, len(dataloader), best_mAP))
        metric_table.inner_footing_row_border = True
        logger.print_and_write('{}\n'.format(metric_table.table))

        # Clear CUDA memory after each batch
        torch.cuda.empty_cache()

    scheduler.step()

    # Save checkpoints
    states = {
        'epoch': epoch + 1,
        'model': opt.model,
        'state_dict': model.module.state_dict() if ngpu > 1 else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_mAP': best_mAP,
    }

    save_file_path = os.path.join(opt.checkpoint_path, 'last.pth')
    torch.save(states, save_file_path)

    if epoch % opt.checkpoint_interval == 0:
        save_file_path = os.path.join(opt.checkpoint_path, f'epoch_{epoch}.pth')
        torch.save(states, save_file_path)
def train(model, optimizer, scheduler, dataloader, epoch, opt, logger, best_mAP=0):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() and opt.gpu else "cpu")
    ngpu = torch.cuda.device_count() if device.type == "cuda" else 1

    for i, (images, targets, indexes) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()

        if targets is None or targets.numel() == 0:
            continue

        if targets.dim() == 1:
            targets = targets.unsqueeze(0)

        images = images.to(device)
        targets = targets.to(device)
        indexes = indexes.to(device)

        rep_targets = [targets.unsqueeze(0) for _ in range(ngpu)]
        rep_targets = torch.cat(rep_targets, dim=0).to(device)

        loss, detections = model(images, rep_targets, indexes)
        if ngpu > 1:
            loss = loss.sum()

        loss.backward()
        optimizer.step()

        if ngpu > 1:
            yolo_layers = model.module.yolo_layers
        else:
            yolo_layers = model.yolo_layers

        metric_keys = yolo_layers[0].metrics.keys()
        metric_table_data = [['Metrics'] + [f'YOLO Layer {i}' for i in range(len(yolo_layers))]]
        for key in metric_keys:
            row = [key] + [f"{yl.metrics.get(key, 0):.6f}" for yl in yolo_layers]
            metric_table_data.append(row)
        metric_table_data.append(['Total loss', f'{loss.item():.6f}'] + [''] * (len(yolo_layers) - 1))

        metric_table = AsciiTable(
            metric_table_data,
            title=f'[Epoch {epoch}/{opt.num_epochs}, Batch {i}/{len(dataloader)}, Current best mAP {best_mAP:.4f}]'
        )
        metric_table.inner_footing_row_border = True
        logger.print_and_write(f'{metric_table.table}\n')

    scheduler.step()

    state = {
        'epoch': epoch + 1,
        'model': opt.model,
        'state_dict': model.module.state_dict() if ngpu > 1 else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_mAP': best_mAP,
    }

    # Save checkpoint as .pt file
    torch.save(state, os.path.join(opt.checkpoint_path, 'last.pt'))

    if epoch % opt.checkpoint_interval == 0:
        torch.save(state, os.path.join(opt.checkpoint_path, f'epoch_{epoch}.pt'))
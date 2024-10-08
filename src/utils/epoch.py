import torch
from tqdm import tqdm
import wandb
from torch import nn
import torch.nn
from ..config import config

device = torch.device(config['device'])


def extract(data, *keys):
    return [data[key.upper()].to(device) for key in keys]


class MTLLoss(nn.Module):
    def __init__(self):
        super(MTLLoss, self).__init__()
        self.loss_fn = nn.HuberLoss(reduction='none')

    def forward(self, epoch, preds, data):
        loss = None
        gts = {}
        mask = extract(data, 'GT_MASK')[0]
        for key, pred_v in preds.items():
            gt_v = extract(data, key)[0]
            gts[key] = gt_v
            if len(pred_v.shape) == 3:
                new_pred_v = pred_v[~mask, :]
                new_gt_v = gt_v[~mask, :]
            else:
                new_pred_v = pred_v[~mask]
                new_gt_v = gt_v[~mask]
            loss_1 = self.loss_fn(new_pred_v, new_gt_v)  # [B, L, C]
            if loss is None:
                loss = torch.mean(loss_1) / torch.mean(loss_1).detach()
            else:
                loss = loss + torch.mean(loss_1) / torch.mean(loss_1).detach()
        gts['GT_MASK'], gts['KAM_MASK'] = extract(data, 'GT_MASK', 'KAM_MASK')
        gts['GT_MASK'] = torch.logical_or(gts['GT_MASK'], gts['KAM_MASK'])
        return loss, gts


def train_test_epoch(_model, data_loader, _epoch, wandb_flag, metric, is_train=True, optimizer=None):
    loss_instance = MTLLoss()
    with tqdm(data_loader, unit="batch", ncols=80) as t_epoch:
        if is_train:
            torch.enable_grad()
            _model.train()
        else:
            torch.no_grad()
            _model.eval()
        for i, data in enumerate(t_epoch):
            t_epoch.set_description(f"{'Train' if is_train else 'Test'} Epoch {_epoch}")
            preds = _model(data)
            loss_v, gts = loss_instance(_epoch, preds, data)
            if is_train:
                optimizer.zero_grad()
                loss_v.backward()
                optimizer.step()

            else:
                gts['MIN_MAX'] = data['MIN_MAX']
                for key in ['KAM', 'force']:
                    if key not in preds:
                        continue
                    if key == 'force':
                        gts['MIN_MAX'][key][0] = gts['MIN_MAX'][key][0].to(device).unsqueeze(1)
                        gts['MIN_MAX'][key][1] = gts['MIN_MAX'][key][1].to(device).unsqueeze(1)
                    else:
                        gts['MIN_MAX'][key][0] = gts['MIN_MAX'][key][0].to(device).unsqueeze(1).unsqueeze(2)
                        gts['MIN_MAX'][key][1] = gts['MIN_MAX'][key][1].to(device).unsqueeze(1).unsqueeze(2)
                    gts[key] = gts[key] / (gts['MIN_MAX'][key][1] - gts['MIN_MAX'][key][0])
                    preds[key] = preds[key] / (gts['MIN_MAX'][key][1] - gts['MIN_MAX'][key][0])
                metric.update(preds, gts)

            torch.cuda.empty_cache()
        if not is_train:
            metric_r = metric.compute(preds.keys(), is_train)
        # scheduler.step()
        if wandb_flag and not is_train:
            wandb.log({**metric_r}, commit=not is_train)
            metric.reset()
    if not is_train:
        return metric_r

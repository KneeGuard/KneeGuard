import torch
import torchmetrics


class NRMSE(torchmetrics.Metric):
    def __init__(self):
        super(NRMSE, self).__init__()

        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('n', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('global_max', default=torch.tensor(0.0), dist_reduce_fx='max')

    def update(self, pred, target):
        self.sum += torch.sqrt(torch.sum((pred - target) ** 2) / pred.numel())
        self.n += 1

    def compute(self):
        return self.sum / self.n


class Metric:
    def __init__(self, device, labels=None):
        self.device = device
        if labels is None:
            self.labels = labels = {
                'force': ['vasint', 'bflh', 'gasmed'],
                'kam': ['kam'],
                'mtu': ['bflh_mtu_l', 'gasmed_mtu_l', 'vasint_mtu_l', 'bflh_mtu_v', 'gasmed_mtu_v', 'vasint_mtu_v'],
                'k': ['hip_flexion', 'hip_adduction', 'hip_rotation', 'knee_angle', 'ankle_angle'],
                'emg': [f'emg {i}' for i in range(14)]
            }

        self.metric_instances = {}
        for label_class in labels:
            self.metric_instances[label_class] = []
            for label in labels[label_class]:
                nrmse = NRMSE().to(device)
                nrmse.name = label + '_NRMSE'
                self.metric_instances[label_class].append([nrmse])

    def update(self, preds, gts):
        for value_class in preds:
            if value_class not in self.metric_instances:  # should be invalid
                continue
            pred_v = preds[value_class]  # B, L, C
            gt_v = gts[value_class]
            for b in range(pred_v.shape[0]):
                gt_m = gts['GT_MASK']
                for c in range(pred_v.shape[2]):
                    for m in self.metric_instances[value_class][c]:  # R2/NRMSE instance
                        m.update(pred_v[b, ~gt_m[b], c], gt_v[b, ~gt_m[b], c])

    def compute(self, classes, is_train):
        r = {}
        for value_class in classes:
            if value_class not in self.metric_instances:
                continue
            for c in range(len(self.metric_instances[value_class])):
                for m in self.metric_instances[value_class][c]:
                    r[('train_' if is_train else '') + m.name] = m.compute().cpu().detach()
        return r

    def reset(self):
        for value_class in self.metric_instances:
            for c in range(len(self.metric_instances[value_class])):
                for m in self.metric_instances[value_class][c]:
                    m.reset()

import scipy.signal
import torch
from torch.utils.data import Dataset
import numpy as np
from ..utils import preprocess as preprocess
from torch.nn.utils.rnn import pad_sequence
import re
import os


def KAM_test(ts):  # [T, ]
    # False on real KAM, True on interpolated part
    mask_pos = (np.abs(np.diff(ts, n=2)) < 1e-5).astype(bool)
    mask_pos = np.array([False, False, *mask_pos]) | np.array([*mask_pos, False, False])
    return mask_pos


class PreparedCycleCutDataset(Dataset):
    def __init__(self, datafile_list, config, dtype):
        self.emg_data = []
        self.imu_data = []
        self.force_data = []
        self.KAM_data = []
        self.k_data = []
        self.mtu_data = []
        self.info = []
        self.emg_mask = []
        self.imu_mask = []
        self.gt_mask = []
        self.KAM_mask = []
        self.min_max_table = {

        }

        for i, file in enumerate(datafile_list):
            name, speed, type_, time = os.path.basename(file).split('_')
            with np.load(file, allow_pickle=True) as data:
                emg, imu, KAM, force, k, mtu, info = \
                    data['EMG'], data['IMU'], data['KAM'] / (data['BW'] * data['BH']), \
                    data['FORCE'] / data['BW'], data['KINEMATIC'], data['MTU'] * 100, \
                    np.array([[name, speed, re.sub('\d*', '', type_), type_, time[:-4]]])

                l = force.shape[0]
                if f'{name}_{info[0][2]}' not in self.min_max_table:
                    self.min_max_table[f'{name}_{info[0][2]}'] = {
                        'KAM': [np.min(KAM), np.max(KAM)],
                        'force': [np.min(force, 0), np.max(force, 0)]
                    }
                else:
                    self.min_max_table[f'{name}_{info[0, 2]}']['KAM'][0] = np.minimum(np.min(KAM), self.min_max_table[
                        f'{name}_{info[0, 2]}']['KAM'][0])
                    self.min_max_table[f'{name}_{info[0, 2]}']['KAM'][1] = np.maximum(np.max(KAM), self.min_max_table[
                        f'{name}_{info[0, 2]}']['KAM'][1])
                    self.min_max_table[f'{name}_{info[0, 2]}']['force'][0] = np.minimum(np.min(force, 0),
                                                        self.min_max_table[f'{name}_{info[0, 2]}']['force'][0])
                    self.min_max_table[f'{name}_{info[0, 2]}']['force'][1] = np.maximum(np.max(force, 0),
                                                        self.min_max_table[f'{name}_{info[0, 2]}']['force'][1])

                cycle_idx = preprocess.gait_seg(imu[:, 16])
                for idx0, idx1 in zip(cycle_idx[:-1], cycle_idx[1:]):
                    if idx1 // 2 - idx0 // 2 > 200:
                        # fail case
                        continue

                    emg_c = emg[idx0 // 2 * 20:idx1 // 2 * 20, :]
                    emg_spec = np.abs(
                        scipy.signal.stft(emg_c.T, fs=2000, nperseg=64, noverlap=59, nfft=256, padded=False)[2])

                    self.emg_data.append(torch.tensor(emg_spec[:, :64, :-1]).permute(2, 1, 0).to(dtype=dtype))
                    self.imu_data.append(torch.tensor(imu[idx0 // 2 * 2:idx1 // 2 * 2, :], dtype=dtype))
                    self.force_data.append(torch.tensor(force[idx0 // 2:idx1 // 2, :], dtype=dtype))
                    self.KAM_data.append(torch.tensor(KAM[idx0 // 2:idx1 // 2], dtype=dtype).unsqueeze(-1))
                    self.mtu_data.append(torch.tensor(mtu[idx0 // 2:idx1 // 2, :], dtype=dtype))
                    self.k_data.append(torch.tensor(k[idx0 // 2:idx1 // 2, :], dtype=dtype))
                    self.imu_mask.append(torch.zeros(idx1 // 2 * 2 - idx0 // 2 * 2, dtype=torch.bool))
                    self.gt_mask.append(torch.zeros(idx1 // 2 - idx0 // 2, dtype=torch.bool))
                    self.KAM_mask.append(
                        torch.tensor(KAM_test(KAM[idx0 // 2:idx1 // 2])).to(dtype=dtype))

                    self.info.append(info)


class CycleCutDataset(Dataset):

    def __init__(self, dataset_list: list[PreparedCycleCutDataset], config, dtype, device):
        self.emg_data = []
        self.imu_data = []
        self.force_data = []
        self.KAM_data = []
        self.k_data = []
        self.mtu_data = []
        self.emg_mask = []
        self.imu_mask = []
        self.gt_mask = []
        self.KAM_mask = []
        self.info = []
        self.device = device
        self.min_max_table = {

        }

        for preparedCycleDataset in dataset_list:
            self.emg_data.extend(preparedCycleDataset.emg_data)
            self.imu_data.extend(preparedCycleDataset.imu_data)
            self.force_data.extend(preparedCycleDataset.force_data)
            self.KAM_data.extend(preparedCycleDataset.KAM_data)
            self.k_data.extend(preparedCycleDataset.k_data)
            self.mtu_data.extend(preparedCycleDataset.mtu_data)
            self.emg_mask.extend(preparedCycleDataset.emg_mask)
            self.imu_mask.extend(preparedCycleDataset.imu_mask)
            self.gt_mask.extend(preparedCycleDataset.gt_mask)
            self.KAM_mask.extend(preparedCycleDataset.KAM_mask)
            self.info.extend(preparedCycleDataset.info)
            self.min_max_table.update(preparedCycleDataset.min_max_table)

        self.emg_data = pad_sequence(self.emg_data, batch_first=True, padding_value=0.).permute(0, 3, 1, 2)
        self.imu_data = pad_sequence(self.imu_data, batch_first=True, padding_value=0.)
        self.force_data = pad_sequence(self.force_data, batch_first=True, padding_value=0.)
        self.KAM_data = pad_sequence(self.KAM_data, batch_first=True, padding_value=0.)
        self.mtu_data = pad_sequence(self.mtu_data, batch_first=True, padding_value=0.)
        self.k_data = pad_sequence(self.k_data, batch_first=True, padding_value=0.)
        # self.emg_mask = pad_sequence(self.emg_mask, batch_first=True, padding_value=1.)
        self.imu_mask = pad_sequence(self.imu_mask, batch_first=True, padding_value=1.)
        self.gt_mask = pad_sequence(self.gt_mask, batch_first=True, padding_value=1.)
        self.KAM_mask = pad_sequence(self.KAM_mask, batch_first=True, padding_value=1.)

    def __getitem__(self, idx):
        return {
            'EMG': self.emg_data[idx], 'IMU': self.imu_data[idx], 'KAM': self.KAM_data[idx],
            'FORCE': self.force_data[idx], 'MTU': self.mtu_data[idx],
            'K': self.k_data[idx], 'IMU_MASK': self.imu_mask[idx], 'MIN_MAX': self.get_min_max(idx),
            'GT_MASK': self.gt_mask[idx], 'KAM_MASK': self.KAM_mask[idx]}

    def __len__(self):
        return self.emg_data.shape[0]

    def get_info(self, idx):
        return self.info[idx]

    def get_min_max(self, idx):
        info = self.get_info(idx)
        return self.min_max_table[f'{info[0][0]}_{info[0][2]}']



from typing import Dict

from .predictor import *
from .extractor import *
from ..utils.epoch import extract
from ..dataset.muscle_dataset import *


class MultiTaskModel(nn.Module):
    def __init__(self, config):
        super(MultiTaskModel, self).__init__()

        self.config = config
        params = config
        d_model = params['transformer_size']
        ffn_hidden = params['cross_mod_ffn_hidden']
        dropout = params['transformer_dropout']
        n_head = params['transformer_n_head']
        dim_forward = params['transformer_dim_forward']
        self.pre_fused_model = MuscleFormer(config)
        self.post_fused_model = MultitaskLinearPredictor(config)

        self.vas_pos = LearnedPositionEncoding(d_model * 2)
        self.bf_pos = LearnedPositionEncoding(d_model * 2)
        self.gas_pos = LearnedPositionEncoding(d_model * 2)
        self.kam_pos = LearnedPositionEncoding(d_model * 2)

        self.fused_fext_vas_imu = CrossModalEncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head,
                                                         drop_prob=dropout)
        self.fused_fext_vas_imu2 = CrossModalEncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head,
                                                          drop_prob=dropout)
        self.fused_fext_vas_emg = CrossModalEncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head,
                                                         drop_prob=dropout)
        self.fused_fext_vas = nn.TransformerEncoderLayer(d_model=2 * d_model, nhead=n_head,
                                                         dim_feedforward=2 * dim_forward, dropout=dropout,
                                                         batch_first=True)
        self.fused_fext_bf_imu = CrossModalEncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head,
                                                        drop_prob=dropout)
        self.fused_fext_bf_imu2 = CrossModalEncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head,
                                                         drop_prob=dropout)
        self.fused_fext_bf_emg = CrossModalEncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head,
                                                        drop_prob=dropout)
        self.fused_fext_bf = nn.TransformerEncoderLayer(d_model=2 * d_model, nhead=n_head,
                                                        dim_feedforward=2 * dim_forward, dropout=dropout,
                                                        batch_first=True)
        self.fused_fext_gas_imu = CrossModalEncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head,
                                                         drop_prob=dropout)
        self.fused_fext_gas_imu2 = CrossModalEncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head,
                                                          drop_prob=dropout)
        self.fused_fext_gas_emg = CrossModalEncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head,
                                                         drop_prob=dropout)
        self.fused_fext_gas = nn.TransformerEncoderLayer(d_model=2 * d_model, nhead=n_head,
                                                         dim_feedforward=2 * dim_forward, dropout=dropout,
                                                         batch_first=True)
        self.fused_fext_kam_imu = CrossModalEncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head,
                                                         drop_prob=dropout)
        self.fused_fext_kam_emg = CrossModalEncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head,
                                                         drop_prob=dropout)
        self.fused_fext_kam = nn.TransformerEncoderLayer(d_model=2 * d_model, nhead=n_head,
                                                         dim_feedforward=4 * dim_forward, dropout=dropout,
                                                         batch_first=True)

    def forward(self, data: Dict[str, torch.Tensor]):
        # if self.cycle_cut:
        emg_data, imu_data, imu_mask, gt_mask = extract(data, 'emg', 'imu', 'imu_mask', 'gt_mask')

        emg_f_vas, emg_f_bf, emg_f_gas, emg_f_kam, imu_f, imu_f2_vas, imu_f2_bf, imu_f2_gas, imu_f2_kam = self.pre_fused_model(
            emg_data, imu_data, imu_mask, gt_mask)
        fe_vas = torch.cat([
            self.fused_fext_vas_imu(x=imu_f2_vas, q=emg_f_vas),
            self.fused_fext_vas_emg(x=emg_f_vas, q=imu_f2_vas),
        ], dim=-1)  # 64
        fe_vas = fe_vas + self.vas_pos(fe_vas)
        fe_vas = self.fused_fext_vas(fe_vas, src_key_padding_mask=gt_mask)

        fe_bf = torch.cat([
            self.fused_fext_bf_imu(x=imu_f2_bf, q=emg_f_bf),
            self.fused_fext_bf_emg(x=emg_f_bf, q=imu_f2_bf),
        ], dim=-1)
        fe_bf = fe_bf + self.bf_pos(fe_bf)
        fe_bf = self.fused_fext_bf(fe_bf, src_key_padding_mask=gt_mask)

        fe_gas = torch.cat([
            self.fused_fext_gas_imu(x=imu_f2_gas, q=emg_f_gas),
            self.fused_fext_gas_emg(x=emg_f_gas, q=imu_f2_gas),
        ], dim=-1)
        fe_gas = fe_gas + self.bf_pos(fe_gas)
        fe_gas = self.fused_fext_gas(fe_gas, src_key_padding_mask=gt_mask)

        fe_kam = torch.cat([
            self.fused_fext_kam_imu(x=imu_f, q=emg_f_kam),
            self.fused_fext_kam_emg(x=imu_f2_kam, q=emg_f_kam),
        ], dim=-1)
        fe_kam = fe_kam + self.kam_pos(fe_kam)
        fe_kam = self.fused_fext_kam(fe_kam, src_key_padding_mask=gt_mask)

        force_pred, mtu_pred, kam_pred, k_pred = self.post_fused_model(
            [imu_f2_vas, imu_f2_bf, imu_f2_gas, imu_f2_vas, imu_f2_bf, imu_f2_gas], [fe_vas, fe_bf, fe_gas], imu_f2_kam,
            fe_kam)

        return {
            'force': force_pred,  # 3
            'mtu': mtu_pred,  # 3
            'KAM': kam_pred,  # 1
            'k': k_pred,  # 8
        }


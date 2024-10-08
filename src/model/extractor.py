from .block import *


class MuscleFormer(nn.Module):
    def __init__(self, config):
        super(MuscleFormer, self).__init__()

        self.config = config

        params = config
        d_model = params['transformer_size']
        dropout = params['transformer_dropout']
        n_head = params['transformer_n_head']
        dim_forward = params['transformer_dim_forward']

        self.down_conv = nn.ModuleList([nn.Sequential(
            nn.Conv2d(64, 64, (3, 15), stride=(1, 2), padding=(1, 7), padding_mode='circular'),
            nn.SELU(),
            nn.Conv2d(64, 32, (3, 15), stride=(1, 1), padding=(1, 7), padding_mode='circular'),
            nn.SELU(),
            nn.Conv2d(32, 16, (3, 15), stride=(1, 2), padding=(1, 7), padding_mode='circular'),
            nn.SELU()
        ) for _ in range(3)])

        self.vas_sa = SpatialAttention(4)
        self.bf_sa = SpatialAttention(4)
        self.gas_sa = SpatialAttention(6)
        self.kam_sa = SpatialAttention(14)

        self.vas_linear = nn.Linear(64, d_model)
        self.bf_linear = nn.Linear(64, d_model)
        self.gas_linear = nn.Linear(96, d_model)
        self.kam_linear = nn.Linear(224, d_model)

        self.emg_pos_encode = LearnedPositionEncoding(d_model)

        self.vas_fext = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_forward,
                                                   batch_first=True)
        self.bf_fext = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_forward,
                                                  batch_first=True)
        self.gas_fext = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_forward,
                                                   batch_first=True)
        self.kam_fext = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_forward,
                                                   batch_first=True)

        self.emg_norm = nn.LayerNorm(d_model)

        self.imu_embed1 = nn.Linear(config['num_imu_channel'] // 2, d_model // 2)
        self.imu_embed2 = nn.Linear(config['num_imu_channel'] // 2, d_model // 2)

        self.vas_pos_imu = LearnedPositionEncoding(d_model)
        self.bf_pos_imu = LearnedPositionEncoding(d_model)
        self.gas_pos_imu = LearnedPositionEncoding(d_model)
        self.kam_pos_imu = LearnedPositionEncoding(d_model)

        self.imu_fext = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_forward,
                                                   batch_first=True)

        self.imu_fext_2_vas = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_forward,
                                                         batch_first=True)
        self.imu_fext_2_bf = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_forward,
                                                        batch_first=True)
        self.imu_fext_2_gas = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_forward,
                                                         batch_first=True)
        self.imu_fext_2_kam = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_forward,
                                                         batch_first=True)

        self.imu_down = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, padding=2),
            nn.SELU(),
        )

        self.imu_pos = LearnedPositionEncoding(d_model)
        self.imu_norm_upper = nn.LayerNorm(d_model // 2)
        self.imu_norm_lower = nn.LayerNorm(d_model // 2)
        self.imu_pos_encode_upper = LearnedPositionEncoding(d_model // 2)
        self.imu_pos_encode_lower = LearnedPositionEncoding(d_model // 2)

        self.imu_fext1 = CrossModalEncoderLayer(d_model=d_model // 2, n_head=n_head, ffn_hidden=dim_forward,
                                                drop_prob=dropout)
        self.imu_fext2 = CrossModalEncoderLayer(d_model=d_model // 2, n_head=n_head, ffn_hidden=dim_forward,
                                                drop_prob=dropout)

        self.act = nn.SELU()

    def forward(self, emg_data, imu_data, imu_mask, gt_mask):
        # emg_data [B, C(14), 4*L, F(64)] => [B, 64, C, 4*L] => [B, 64, C, L]
        vas_embed = self.down_conv[0](emg_data[:, 10:, :, :].permute(0, 3, 1, 2))
        bf_embed = self.down_conv[1](emg_data[:, 6:10, :, :].permute(0, 3, 1, 2))
        gas_embed = self.down_conv[2](emg_data[:, :6, :, :].permute(0, 3, 1, 2))
        kam_embed = torch.cat([vas_embed, bf_embed, gas_embed], dim=2).detach()

        vas_sa = self.vas_sa(vas_embed)
        bf_sa = self.bf_sa(bf_embed)
        gas_sa = self.gas_sa(gas_embed)
        kam_sa = self.kam_sa(kam_embed)
        # [B, 1, C, L] * [B, 32, C, L]
        vas_embed = vas_sa * vas_embed  # [B, 16, 4, L]
        bf_embed = bf_sa * bf_embed  # [B, 16, 4, L]
        gas_embed = gas_sa * gas_embed  # [B, 16, 6, L]
        kam_embed = kam_sa * kam_embed

        vas_embed = self.emg_norm(self.vas_linear(
            vas_embed.reshape(vas_embed.shape[0], vas_embed.shape[3], vas_embed.shape[1] * vas_embed.shape[2])))
        bf_embed = self.emg_norm(self.bf_linear(
            bf_embed.reshape(bf_embed.shape[0], bf_embed.shape[3], bf_embed.shape[1] * bf_embed.shape[2])))
        gas_embed = self.emg_norm(self.gas_linear(
            gas_embed.reshape(gas_embed.shape[0], gas_embed.shape[3], gas_embed.shape[1] * gas_embed.shape[2])))
        kam_embed = self.emg_norm(self.kam_linear(
            kam_embed.reshape(kam_embed.shape[0], kam_embed.shape[3], kam_embed.shape[1] * kam_embed.shape[2])))

        vas_embed = vas_embed + self.emg_pos_encode(vas_embed)
        bf_embed = bf_embed + self.emg_pos_encode(bf_embed)
        gas_embed = gas_embed + self.emg_pos_encode(gas_embed)
        kam_embed = kam_embed + self.emg_pos_encode(kam_embed)

        emg_vas_f = self.vas_fext(vas_embed, src_key_padding_mask=gt_mask)
        emg_bf_f = self.bf_fext(bf_embed, src_key_padding_mask=gt_mask)
        emg_gas_f = self.gas_fext(gas_embed, src_key_padding_mask=gt_mask)
        emg_kam_f = self.kam_fext(kam_embed, src_key_padding_mask=gt_mask)

        print(imu_data.shape)
        imu_embed1 = self.imu_norm_upper(self.imu_embed1(imu_data[:, :, :13]))
        imu_embed2 = self.imu_norm_lower(self.imu_embed2(imu_data[:, :, 13:]))

        imu_embed1 = imu_embed1 + self.imu_pos_encode_upper(imu_embed1)
        imu_embed2 = imu_embed2 + self.imu_pos_encode_lower(imu_embed2)

        imu_embed = torch.cat([
            self.imu_fext1(x=imu_embed1, q=imu_embed2),
            self.imu_fext2(x=imu_embed2, q=imu_embed1),
        ], dim=-1)

        imu_embed = self.act(self.imu_down(imu_embed.permute(0, 2, 1))).permute(0, 2, 1)
        imu_embed = imu_embed + self.imu_pos(imu_embed)

        imu_f = self.imu_fext(imu_embed, src_key_padding_mask=gt_mask)

        imu_f_vas = imu_f + self.vas_pos_imu(imu_f)
        imu_f_bf = imu_f + self.bf_pos_imu(imu_f)
        imu_f_gas = imu_f + self.gas_pos_imu(imu_f)
        imu_f_kam = imu_f + self.kam_pos_imu(imu_f)

        imu_f2_vas = self.imu_fext_2_vas(imu_f_vas, src_key_padding_mask=gt_mask)
        imu_f2_bf = self.imu_fext_2_bf(imu_f_bf, src_key_padding_mask=gt_mask)
        imu_f2_gas = self.imu_fext_2_gas(imu_f_gas, src_key_padding_mask=gt_mask)
        imu_f2_kam = self.imu_fext_2_kam(imu_f_kam, src_key_padding_mask=gt_mask)

        return emg_vas_f, emg_bf_f, emg_gas_f, emg_kam_f, imu_f, imu_f2_vas, imu_f2_bf, imu_f2_gas, imu_f2_kam


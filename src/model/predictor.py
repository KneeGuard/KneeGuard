from .block import *


class MultitaskLinearPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        param = config
        fused_fe_size = param['transformer_size']*2
        kam_fe_size = param['transformer_size']*2
        imu_fe_size = param['transformer_size']
        hidden_size1 = param['fc_hidden_size1']
        hidden_size2 = param['fc_hidden_size2']
        drop_rate = param['fc_dropout']

        self.force_predictor = nn.ModuleList([FCPredictor(
            fused_fe_size,
            hidden_size1,
            hidden_size2,
            1,
            drop_rate,
        ) for _ in range(len(config['force_channel']))])

        self.mtu_predictor = nn.ModuleList([FCPredictor(
            imu_fe_size,
            hidden_size1,
            hidden_size2,
            1,
            drop_rate,
        ) for _ in range(len(config['mtu_channel']))])

        self.kam_predictor = FCPredictor(
            kam_fe_size,
            hidden_size1,
            hidden_size2,
            1,
            drop_rate,
        )

        self.k_predictors = nn.ModuleList([FCPredictor(
            imu_fe_size,
            hidden_size1,
            hidden_size2,
            1,
            drop_rate,
        ) for _ in range(len(config['k_channel']))])

        self.grf_predictors = nn.ModuleList([FCPredictor(
            kam_fe_size,
            hidden_size1,
            hidden_size2,
            1,
            drop_rate,
        ) for _ in range(len(config['grf_channel']))])

    def forward(self, fe_mtu, fe_force, fe_k, fe_kam):
        return torch.cat([pred(fe_force[i]) for i, pred in enumerate(self.force_predictor)], dim=-1), \
            torch.cat([pred(fe_mtu[i]) for i, pred in enumerate(self.mtu_predictor)], dim=-1), \
            self.kam_predictor(fe_kam), \
            torch.cat([predictor(fe_k) for predictor in self.k_predictors], dim=-1)
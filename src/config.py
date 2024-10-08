config = {
    'wandb_flag': True,
    'wandb_api_key': None,
    'wandb_group': "test",
    'transformer_n_head': 4,
    'cross_mod_ffn_hidden': 144,
    'transformer_dim_forward': 256,
    'transformer_dropout': 0.1,
    'transformer_size': 64,
    'fc_hidden_size1': 32,
    'fc_hidden_size2': 16,
    'fc_dropout': 0.0,


    'device': 'cpu',
    'torch_dtype': 'torch.float32',
    'seed': 1111,
    'save_prediction': False,
    'save_path': None,
    'model_path': None,
    'datafile_path': './data/',
    'dataset_path': './data/',
    'num_epoch': 100,
    'num_model_save_epoch': 100,
    'train_batch_size': 32,
    'test_batch_size': 32,
    'optimizer': 'torch.optim.Adam',
    'loss_fn': 'torch.nn.HuberLoss',
    'lr': 1e-3,

    'L': 256,

    # Subject ID
    'SID': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14',
            'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28',
            'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38'],

}
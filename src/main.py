import gc
import wandb

from .config import config
from torch.utils.data import DataLoader
from .utils.epoch import *
from .dataset.muscle_dataset import *
from .model.arch import *
from .metric.metrics import Metric
import torch.optim

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def prepareDatasets(test_id):
    index_list = [file for file in os.listdir(config['datafile_path']) if 'npz' in file]
    device = torch.device(config['device'])
    dtype = eval(config['torch_dtype'])
    train_dataset_list = []
    test_dataset = []
    for sid in config['SID']:
        if not os.path.exists(f"{config['dataset_path']}{sid}.pt"):
            data_file_list = [config['datafile_path'] + file for file in index_list
                              if file.split('_')[0] == sid]
            if len(data_file_list) > 0:
                dataset = PreparedCycleCutDataset(data_file_list, config, dtype)
                torch.save(dataset, f"{config['dataset_path']}{sid}.pt")
            else:
                continue
        else:
            dataset = torch.load(f"{config['dataset_path']}{sid}.pt")
        if sid != test_id:
            train_dataset_list.append(dataset)
        else:
            test_dataset.append(dataset)
    train_data = CycleCutDataset(train_dataset_list, config, dtype, device)
    test_data = CycleCutDataset(test_dataset, config, dtype, device)
    del train_dataset_list, test_dataset
    return train_data, test_data


def main():
    wandb_flag = config['wandb_flag']
    device = torch.device(config['device'])
    torch.manual_seed(config['seed'])
    dtype = eval(config['torch_dtype'])
    torch.set_default_dtype(dtype)
    wandb.login()
    run = wandb.init(project="muscle", entity='muscle', group=config['wandb_group'],
                     mode="online" if wandb_flag else "offline", save_code=True,
                     config=config, settings=wandb.Settings(code_dir="."))
    run.name = f"LOO_{wandb.config['SID']}"

    # The test ID
    train_data, test_data = prepareDatasets(wandb.config['SID'])

    train_dataloader = DataLoader(train_data, batch_size=config['train_batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=config['test_batch_size'])

    model = MultiTaskModel(config).to(device)
    optimizer = eval(config['optimizer'])(model.parameters(), lr=float(config['lr']))
    metric = Metric(device)

    for epoch in range(1, config['num_epoch'] + 1):
        train_test_epoch(model, train_dataloader, epoch,
                         wandb_flag, metric, True, optimizer)
        train_test_epoch(model, test_dataloader, epoch,
                         wandb_flag, metric, False)
        if epoch % config['num_model_save_epoch'] == 0:
            torch.save(model,
                       config['model_path'] + config['wandb_group'] + f"_fold{wandb.config['SID']}_epoch{epoch}")

    gc.collect()
    torch.cuda.empty_cache()
    wandb.finish()


if __name__ == '__main__':
    pass
    # os.environ["WANDB_API_KEY"] = config['wandb_api_key']

    # sweep_config = {
    #     "method": "grid",
    #     "metric": {"goal": "minimize", "name": "vasint_NRMSE"},
    #     "parameters": {
    #         'SID': {'values': config['SID']},
    #     },
    # }

    # sweep_id = wandb.sweep(sweep=sweep_config, project="muscle", entity='muscle')
    # wandb.agent(sweep_id, function=main)

    # main()

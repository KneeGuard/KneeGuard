import torch
from torch.utils.data import DataLoader

from .main import prepareDatasets
from .utils.epoch import train_test_epoch
from .metric.metrics import Metric
from .config import *
import argparse


def test(SID):
    device = torch.device(config['device'])
    torch.manual_seed(config['seed'])
    dtype = eval(config['torch_dtype'])
    torch.set_default_dtype(dtype)
    _, test_data = prepareDatasets(SID)
    test_dataloader = DataLoader(test_data, batch_size=config['test_batch_size'])

    model = torch.load(f'./model/{SID}.pt', map_location=device)
    metric = Metric(device)

    result = train_test_epoch(model, test_dataloader, 1, False, metric, False)

    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    import warnings

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    parser.add_argument('-S', '--subject', choices=['S1', 'S27'], type=str, help='Input testing subject', default='S1')

    args = parser.parse_args()
    test(args.subject)

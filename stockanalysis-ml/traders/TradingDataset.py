from datasets.Common import TimeSeriesDataset
from models.Common import StandardModel, get_bar_format
from traders.TradingSimulation import TradingSimulation

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class TradingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, inference_model: StandardModel):
        self.df = df

        real_len = inference_model.conf.seq_len
        infer_len = inference_model.conf.out_seq_len
        columns = inference_model.conf.column_names
        self.real_len = real_len

        print(f'> Generating inference data:')
        print(f'Inference model window: {real_len}x{len(columns)}+{infer_len}; Trader window: {real_len}+{infer_len}+{TradingSimulation.state_size()}')

        device = torch.device(inference_model.device)

        # Get batches from the dataframe
        batch_size = 64
        # Output window for the TSDataset is 0
        # since we won't need to cut datapoints from the end to use in loss functions.
        batch_data = DataLoader(TimeSeriesDataset(df, real_len, 0, columns), batch_size)

        bar_format = get_bar_format(len(batch_data), 1)

        # Generate future inferred datapoints
        self.inferences = []
        for i, data in tqdm(enumerate(batch_data), total=len(batch_data), bar_format=bar_format):
            X = torch.Tensor(data["X"]).float().to(device)
            y = inference_model.infer(X, False, False)

            self.inferences.append(y)

        print()

    def __len__(self):
        return len(self.df) - self.real_len + 1

    def __getitem__(self, index):
        real_values = self.df['close'][index: index + self.real_len]
        return { "real": real_values.values, "inference": self.inferences[index] }
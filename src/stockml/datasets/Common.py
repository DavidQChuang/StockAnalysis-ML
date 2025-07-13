import inspect
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils import data
from torch.utils.data.dataset import Dataset

from stockml.datasets.sources import Source, SourceColumn
from stockml.vprint import vprint


@dataclass
class DatasetConfig:
    @classmethod
    def from_dict(cls, env):
        def parse_column_flags(kv):
            return (
                (kv[0], [SourceColumn.from_dict(col_json) for col_json in kv[1]])
                if kv[0] == "include_columns"
                else (kv)
            )

        return cls(**{k: v for k, v in map(parse_column_flags, env.items()) if k in inspect.signature(cls).parameters})  # type: ignore

    # Model I/O window sizes
    # These values will be copied from the model JSON,
    # so the values of these don't matter and they don't need to be defined in 'dataset'.
    seq_len: int = 0
    out_seq_len: int = 0

    sources: dict[str, dict[str, Any]] = field(default_factory=lambda: {})
    # Fields passed to sources
    resample_intervals: list[str] = field(default_factory=lambda: [])
    indicators: list[dict[str, Any]] = field(default_factory=lambda: [])
    include_columns: list[SourceColumn] = field(default_factory=lambda: [])

    # Other parmaeters for use during inference training
    test_split: float = 0.1
    validation_split: float = 0.2
    batch_size: int = 64

    target: str = "close"

    @classmethod
    def get_source_inherited_keys(cls):
        return ["resample_intervals", "indicators", "include_columns"]

    @classmethod
    def get_source_inherited_values(cls, dataset_json):
        return {key: dataset_json[key] for key in cls.get_source_inherited_keys()}


@dataclass
class DataframeConfig:
    @classmethod
    def from_dict(cls, env):
        return cls(**{k: v for k, v in env.items() if k in inspect.signature(cls).parameters})

    symbol: str = ""
    interval: timedelta = timedelta()


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target="close",
        seq_len=0,
        out_seq_len=0,
        test_split=0.1,
        validation_split=0.2,
        batch_size=64,
        columns: list[str] = None,
        scaled_columns: list[str] = None,
    ):  # type: ignore
        self.df: pd.DataFrame = df
        self._target = target
        self._seq_len = seq_len
        self._out_seq_len = out_seq_len
        self._columns = columns
        self._scaled_columns = scaled_columns
        self._test_split = test_split
        self._validation_split = validation_split
        self._batch_size = batch_size

        self.scaler: StandardScaler | None = None

        if self.columns is None or len(self.columns) == 0:
            raise Exception("Dataset was given no column names to use as input.")

    @property
    def target(self) -> str:
        return self._target

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def out_seq_len(self) -> int:
        return self._out_seq_len

    @property
    def test_split(self) -> float:
        return self._test_split

    @property
    def validation_split(self) -> float:
        return self._validation_split

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def columns(self) -> list[str]:
        return self._columns

    @property
    def scaled_columns(self) -> list[str]:
        return self._scaled_columns

    def get_collate_fn(self, device=None, **tensor_args):
        if device is None or device.startswith("cpu"):
            return lambda batch: {
                "X": torch.Tensor(np.array([item["X"] for item in batch]), **tensor_args).float(),
                "y": torch.Tensor(np.array([item["y"] for item in batch]), **tensor_args).float(),
            }
        else:
            return lambda batch: {
                "X": torch.Tensor(np.array([item["X"] for item in batch]), **tensor_args).to(device),
                "y": torch.Tensor(np.array([item["y"] for item in batch]), **tensor_args).to(device),
            }

    def __len__(self):
        return len(self.df) - self.seq_len - self.out_seq_len + 1

    def __getitem__(self, index) -> dict[str, np.ndarray]:
        # if index < 0 and -index <= self.out_seq_len:
        # raise IndexError('There are not enough output data for this sequence; '
        # 'out_seq_len is greater than the number of remaining data points. '
        # 'Use .get(x) instead if you want to get only inputs.')
        return self.get(index)  # type: ignore ; return type is guaranteed

    def get(self, index) -> dict[str, np.ndarray | None]:
        if index < 0:
            # If index is -1, need special range indexer and output is None.
            if index == -1:
                input = self.df[self.columns][-self.seq_len :]
                output = None
            else:
                input = self.df[self.columns][-self.seq_len + index + 1 : index + 1]
                # if not enough data for out_seq_len, output is None.
                if -index <= self.out_seq_len:
                    output = None
                else:
                    output = self.df[self.target][-index + 1 : -index + 1 + self.out_seq_len].values
        else:
            input = self.df[self.columns][index : index + self.seq_len]
            output = self.df[self.target][index + self.seq_len : index + self.seq_len + self.out_seq_len].values

        return {"X": input.values, "y": output}  # type: ignore ; output should be np.ndarray

    def print_validation_split(self, dataset_len, validation_split):
        train_ratio = 1 - validation_split
        train_data = int(dataset_len * train_ratio)

        print(f"Splitting data at a {train_ratio} ratio: {train_data}/{dataset_len - train_data}")

    def scale_dataset(self, scaler: StandardScaler, fit=False):
        dataset = self

        # Store scaler in dataset once used
        if self.scaler is not None:
            raise RuntimeError("Cannot scale a dataset multiple times. This will cause unscaling to be wrong.")

        self.scaler = scaler
        columns_to_scale = self._scaled_columns

        # For display purposes
        preview_columns = dataset.columns
        if "timestamp" in dataset.df.columns:
            preview_columns += ["timestamp"]

        # Actual scaling & fitting
        if fit:
            print("> Scaling and fitting dataset.")
            print("Before: \n", dataset.df[preview_columns][:3], "dtype=", dataset.df["close"].dtype)

            dataset.df[columns_to_scale] = scaler.fit_transform(dataset.df[columns_to_scale])  # type: ignore ; this is matrixlike
        else:
            print("> Scaling dataset.")
            print("Before: \n", dataset.df[preview_columns][:3], "dtype=", dataset.df["close"].dtype)

            if not hasattr(scaler, "mean_"):
                raise RuntimeError(
                    "Scaler must be fitted before being used to scale/unscale input. "
                    "Run TimeSeriesDataset.scale_dataset(scaler, columns_to_scale, fit=True) first."
                )

            dataset.df[columns_to_scale] = scaler.transform(dataset.df[columns_to_scale])  # type: ignore

        print("After: \n", dataset.df[dataset.columns][:3], "dtype=", dataset.df["close"].dtype)
        print(
            "Sanity check (should be equal to first close value): ",
            (dataset.df["close"].iloc[0] * self.scaler.scale_[0] + self.scaler.mean_[0]),
        )  # type: ignore
        print()

        return dataset

    def get_training_data(self, validation_ratio: float, batch_size: int | None = None, pin_memory=False, device="cpu"):
        dataset = self

        if batch_size is None:
            batch_size = self.batch_size

        train_ratio = 1 - validation_ratio
        batch_size = batch_size

        train_data, valid_data = data.random_split(dataset, [train_ratio, validation_ratio])

        # https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-work-in-dataloader
        # If pinning, tensors on CPU remain in non-paged memory.
        # This can speed up calls to Tensor.cuda()
        #   and allows async calls with Tensor.cuda(non_blocking=True).
        if pin_memory:
            print("Pinning")
            train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=False, pin_memory=True)
            valid_dataloader = data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True)
        # If not pinning, use the dataset collate_fn that converts data
        #   in numpy form to tensors on the correct device.
        else:
            print("Unpinned")
            train_dataloader = data.DataLoader(
                train_data, batch_size=batch_size, collate_fn=dataset.get_collate_fn(device=device), shuffle=False
            )
            valid_dataloader = data.DataLoader(
                valid_data, batch_size=batch_size, collate_fn=dataset.get_collate_fn(device=device), shuffle=False
            )

        return train_dataloader, valid_dataloader


class MultisourceTimeSeriesDataset(TimeSeriesDataset):
    def __init__(self, dataset_json: dict[str, Any], verbosity=0):  # type: ignore
        vprint(verbosity, 1, "│├[1/3] > Instantiating MultisourceTimeSeriesDataset.")
        self.conf = DatasetConfig.from_dict(dataset_json)

        # Aggregate sources
        set[str]()
        set[str]()

        vprint(verbosity, 1, f"│├[2/3] > Loading sources [ {', '.join(self.conf.sources.keys())} ].")
        data_sources: list[Source] = []
        for source_name, source_json in self.conf.sources.items():
            if "source_class" not in source_json:
                raise Exception(f"'source_class' cannot be None in source {source_name}.")

            source_class_name = source_json["source_class"]
            SourceClass = Source.get_subclass(source_class_name)

            # Use the dataset's values for these keys
            # unless they're already given inside the source
            copy_values = DatasetConfig.get_source_inherited_values(dataset_json)
            copy_values.update(source_json)  # Overwrites the copy_values with source_json

            if SourceClass is not None:
                source = SourceClass(source_json=copy_values)
                data_sources.append(source)
            else:
                raise Exception(
                    f"Class of 'source_class' ({source_class_name}) could not be found in source {source_name}."
                )

        # Combine dataframes from data sources
        vprint(verbosity, 1, "│└[3/3] > Combining sources.")
        dfs = []
        all_columns_set = set()
        all_columns = []
        all_scaled_columns = []
        for source in data_sources:
            # Get dataframe
            df = source.get_dataframe()
            columns = set(source.columns)
            scaled_columns = set(source.scaled_columns)

            # Make sure no columns overlap
            # TODO: necessary?
            overlapping_cols = all_columns_set.intersection(columns)
            if len(overlapping_cols) != 0:
                raise ValueError(f"Datasources could not be combined due to overlapping columns: {overlapping_cols}")

            # Update set of existing columns
            all_columns_set.update(columns)
            all_columns += columns
            all_scaled_columns += scaled_columns

            # Add dataframe to list of dfs
            dfs.append(df)

        df: pd.DataFrame = pd.concat(dfs, axis=1)

        super().__init__(
            df,
            self.conf.target,
            self.conf.seq_len,
            self.conf.out_seq_len,
            self.conf.test_split,
            self.conf.validation_split,
            self.conf.batch_size,
            all_columns,
            all_scaled_columns,
        )

# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torch.utils.data import get_worker_info
from tqdm.auto import tqdm, trange
import os
import sys
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

from typing import List, Dict, Union, Tuple, Literal

from ..utils.device import pin_memory
from .utils.collate import make_collate_fn, make_merge_collate_fn
from .utils.saved_datasets_manager import SavedDatasetsManager
from .utils.estimate_bytes import compute_sample_bytes
from .utils.io_dtype import optimal_io_dtype

from ..utils import config
from ..utils.misc import zip_dir, unzip

from ..definitions import SAVED_DATASET_OUTPUT_DIR
from ..definitions import TQDM_MININTERVAL

from ..stats import datastats
from .datasets import datasets

import numpy as np

DatasetKindType   = Literal['train', 'val', 'evaluation']
DatasetLengthType = Union[int, List[int]]


class Dataset(TorchDataset):
    """Abstract class for a dataset."""

    def __init__(
        self,
        name:   str,
        size:   int,
        length: DatasetLengthType,
        seed:   int,
        kind:   DatasetKindType
    ):

        if type(self) == Dataset:
            raise Exception('Attempted to instantiate abstract class')

        super().__init__()

        self.name  = name
        self.kind  = kind
        self.seed  = seed

        self.size  = size

        self.sample_seeds = np.random.SeedSequence([self._kind_to_id(kind), seed]).spawn(size)

        if type(length) == int:
            self.min_len = length
            self.max_len = length
        else:
            if len(length) != 2:
                raise Exception("Length list must be [min_len, max_len]")
            self.min_len = length[0]
            self.max_len = length[1]

        self.data = []


    def _kind_to_id(self, kind):
        if kind == 'train':
            return 0
        elif kind == 'val':
            return 1
        elif kind == 'evaluation':
            return 2
        else:
            raise ValueError(f"Kind '{kind}' is not valid")


    @property
    def input_dtype(self):
        return torch.long

    @property
    def output_dtype(self):
        return torch.long

    def __postinit__(self):
        if type(self) == Dataset:
            raise Exception('Attempted to instantiate abstract class')
        self._load()


    def base_seed(self, seed):
        return np.random.SeedSequence([kind_to_id[self.kind], seed]).spawn(1)[0]


    def __len__(self) -> int:
        return self.size


    def _makeitem(self, idx, seed) -> dict:
        raise Exception('Not implemented')

    @property
    def vocab_size(self):
        raise Exception('Not implemented')

    @property
    def pad_id(self):
        raise Exception('Not implemented')

    def __getitem__(self, idx) -> dict:
        if self.data:
            return self.data[idx]
        else:
            sample = self._makeitem(idx, self.sample_seeds[idx])
            io_dtype   = optimal_io_dtype(self.vocab_size)
            sample['inputs']  = torch.tensor(sample['inputs'],  dtype=io_dtype)
            sample['outputs'] = torch.tensor(sample['outputs'], dtype=io_dtype)
            sample['mask']    = torch.tensor(sample['mask'],    dtype=torch.bool)
            sample['idx']     = idx
            return sample


    def get_loader(self, batch_size) -> DataLoader:

        num_workers = config.get('num_workers')

        if self.kind == 'train':
            drop_last = config.get('train_drop_last')
        elif self.kind == 'val':
            drop_last = config.get('val_drop_last')
        elif self.kind == 'evaluation':
            drop_last = config.get('evaluation_drop_last')

        g = torch.Generator().manual_seed(self.seed)

        collate_fn = make_collate_fn(
            self.max_len,
            self.pad_id,
            self.input_dtype,
            self.output_dtype
        )

        return DataLoader(
                self,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                collate_fn=collate_fn,
                persistent_workers=(num_workers > 0),
                generator=g   # controls shuffling & RandomSampler
                )


    def get_max_len(self):
        return self.max_len

    @staticmethod
    def get_stats_kinds():
        return ['length', 'range_dependency']

    @staticmethod
    def get_required_kwargs():
        return []

    def get_iosize(self):
        raise Exception('Not implemented')

    def saved_exists(self):
        path = self._get_saving_dir()
        return os.path.exists(path + '.zip')


    def generate_and_save(self, long_seqs=False, print_header=''):

        # For progress bar description
        _ph = print_header
        _ds_repr = datasets.repr(self.name, self.seed)
        def __part(idx):            return "(part {})".format(idx) if idx else ''
        def __desc(name, idx=None): return _ph + "{} '{}' {}".format(name, _ds_repr, __part(idx))
        def __desc_gen(idx):        return __desc('Generating', idx)
        def __desc_sav(idx):        return __desc('Saving', idx)
        def __desc_fin():           return __desc('Finalising')
        def __desc_zip():           return __desc('Compressing')
        def __desc_csta():          return __desc('Computing stats')
        def __desc_wsta():          return __desc('Saving stats')

        # Helper function
        def save_part(data, part_idx):
            torch.save(data, self._get_saving_path(part_idx)) # Save
            for i in range(len(data)): data[i] = None        # Clear data
            return part_idx + 1                              # Return next part_idx

        # Begin of function body

        saving_dir = self._create_saving_dir()

        # For short sequences it is faster to have a single thread
        num_workers = config.get('num_workers') if long_seqs else 0

        loader = DataLoader(
                self,
                batch_size=1,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                collate_fn=make_merge_collate_fn(),
                persistent_workers=(num_workers > 0),
                )

        data = [None for _ in range(len(self))]
        stats = datastats.instantiate(len(self), self.get_stats_kinds())
        data_bytes = 0
        part_idx = 1

        with trange(
            len(self),
            desc=__desc_gen(part_idx),
            leave=False,
            unit='samples',
            mininterval=TQDM_MININTERVAL
        ) as pb:
            for batch in loader:
                for idx in batch:
                    sample = batch[idx]
                    data[idx] = sample
                    stats.update(sample['stats'])
                    pb.update()
                    data_bytes += compute_sample_bytes(sample)
                if data_bytes > config.get('saved_dataset_part__threshold_bytes'):
                    pb.set_description(__desc_sav(part_idx))
                    part_idx = save_part(data, part_idx)
                    pb.set_description(__desc_gen(part_idx))
                    data_bytes = 0

            if data_bytes > 0:
                pb.set_description(__desc_sav(part_idx))
                part_idx = save_part(data, part_idx)

            pb.set_description(__desc_fin())
            self._write_saving_meta(n_parts = part_idx-1)
            pb.set_description(__desc_zip())
            zip_dir(saving_dir, saving_dir + '.zip')
            self._delete_saving_dir()

            pb.set_description(__desc_csta())
            computed_stats = stats.compute()

            pb.set_description(__desc_wsta())
            self._write_datastats(computed_stats)


    def _load(self):
        if not self.saved_exists():
            raise Exception(f"Dataset {datasets.repr(self, self.seed)} has not been saved: should be in {self._get_saving_dir()}")

        # Read from zip directly
        msg = ''
        with ZipFile(self._get_saving_dir() + '.zip') as z:
            parts = self._read_saving_meta(z)['n_parts']
            self.data = [None for _ in range(len(self))]
            # Iterate over parts
            for part_idx in range(1, parts + 1):
                with z.open(self._get_saving_path_relative(part_idx)) as f:
                    parts_str = " (part {}/{})".format(part_idx, parts) if parts > 1 else ''
                    len_msg = len(msg)
                    msg = "\rLoading dataset '{}'{}...".format(
                        datasets.repr(self, self.seed),
                        parts_str
                    )
                    print(' ' * len_msg, end='\r')
                    print(msg, end='')
                    data_part = torch.load(f, map_location='cpu')
                    if len(data_part) < len(self):
                        raise Exception('Saved dataset has fewer samples than expected')
                    # Assign samples to their slots
                    for idx in range(len(self)):
                        if data_part[idx] is not None:
                            self.data[idx] = data_part[idx]
            print(' ' * len(msg), end='\r')

        print()


    def _write_datastats(self, stats):
        self_repr = datasets.repr(self, self.seed)
        out_path = Path(os.path.join(SAVED_DATASET_OUTPUT_DIR, 'stats__' + self_repr + '.json'))
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    def _write_saving_meta(self, n_parts):
        path = Path(self._get_saving_meta_path())
        meta = {
            'n_parts': n_parts
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _read_saving_meta(self, z):
        path = self._get_saving_meta_path_relative()
        with z.open(path) as f:
            meta = json.load(f)
        return meta

    def _get_saving_dir(self):
        self_repr = datasets.repr(self, self.seed)
        return os.path.join(SAVED_DATASET_OUTPUT_DIR, self_repr)

    def _create_saving_dir(self):
        path = self._get_saving_dir()
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        return path

    def _delete_saving_dir(self):
        path = self._get_saving_dir()
        shutil.rmtree(path)

    def _get_saving_path(self, idx):
        self_repr = datasets.repr(self, self.seed)
        return os.path.join(self._get_saving_dir(), self_repr + '.' + str(idx) + '.pt')

    def _get_saving_path_relative(self, idx):
        self_repr = datasets.repr(self, self.seed)
        return os.path.join(self_repr + '.' + str(idx) + '.pt')

    def _get_saving_meta_path(self):
        self_repr = datasets.repr(self, self.seed)
        return os.path.join(self._get_saving_dir(), self_repr + '.meta.json')

    def _get_saving_meta_path_relative(self):
        self_repr = datasets.repr(self, self.seed)
        return os.path.join(self_repr + '.meta.json')


"""
Dataset classes for the Temple University Hospital (TUH) EEG Corpus and the
TUH Abnormal EEG Corpus. 

Originally taken from braindecode's repository and adapted to fit the TUEP data loading.

Original Authors: Hubert Banville <hubert.jbanville@gmail.com>
         Lukas Gemein <l.gemein@gmail.com>
         Simon Brandt <simonbrandt@protonmail.com>
         David Sabbagh <dav.sabbagh@gmail.com>
         Robin Schirrmeister <robintibor@gmail.com>

License: BSD (3-clause)
"""

from __future__ import annotations
import re
import os
import glob
import warnings
from unittest import mock
from datetime import datetime, timezone
from typing import Iterable
import pandas as pd
import json
import shutil
import numpy as np
import mne
from joblib import Parallel, delayed
from torch.utils.data import Dataset, ConcatDataset

class BaseDataset(Dataset):
    """Returns samples from an mne.io.Raw object along with a target.

    Dataset which serves samples from an mne.io.Raw object along with a target.
    The target is unique for the dataset, and is obtained through the
    `description` attribute.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous data.
    description : dict | pandas.Series | None
        Holds additional description about the continuous signal / subject.
    target_name : str | tuple | None
        Name(s) of the index in `description` that should be used to provide the
        target (e.g., to be used in a prediction task later on).
    transform : callable | None
        On-the-fly transform applied to the example before it is returned.
    """
    def __init__(self, raw, description=None, target_name=None,
                 transform=None):
        self.raw = raw
        self._description = _create_description(description)
        self.transform = transform

        # save target name for load/save later
        self.target_name = self._target_name(target_name)

    def __getitem__(self, index):
        X = self.raw[:, index][0]
        y = None

        if self.target_name == 'epilepsy':
            y = self.description['epilepsy']
        elif self.target_name is not None:
            y = self.description[self.target_name] #backup
        if isinstance(y, pd.Series):
            y = y.to_list()
        if self.transform is not None:
            X = self.transform(X)

        return X, y

    def __len__(self):
        return len(self.raw)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        if value is not None and not callable(value):
            raise ValueError('Transform needs to be a callable.')
        self._transform = value

    @property
    def description(self):
        return self._description

    def set_description(self, description, overwrite=False):
        """Update (add or overwrite) the dataset description.

        Parameters
        ----------
        description: dict | pd.Series
            Description in the form key: value.
        overwrite: bool
            Has to be True if a key in description already exists in the
            dataset description.
        """
        description = _create_description(description)
        for key, value in description.items():
            # if the key is already in the existing description, drop it
            if self._description is not None and key in self._description:
                assert overwrite, (f"'{key}' already in description. Please "
                                   f"rename or set overwrite to True.")
                self._description.pop(key)
        if self._description is None:
            self._description = description
        else:
            self._description = pd.concat([self.description, description])

    def _target_name(self, target_name):
        if target_name is not None and not isinstance(target_name, (str, tuple, list)):
            raise ValueError('target_name has to be None, str, tuple or list')
        if target_name is None:
            return target_name
        else:
            # convert tuple of names or single name to list
            if isinstance(target_name, tuple):
                target_name = [name for name in target_name]
            elif not isinstance(target_name, list):
                assert isinstance(target_name, str)
                target_name = [target_name]
            assert isinstance(target_name, list)
            # check if target name(s) can be read from description
            for name in target_name:
                if name == 'epilepsy':
                    continue
                elif self.description is None or name not in self.description:
                    warnings.warn(f"'{name}' not in description. '__getitem__'"
                                  f"will fail unless an appropriate target is"
                                  f" added to description.", UserWarning)
        # return a list of str if there are multiple targets and a str otherwise
        return target_name if len(target_name) > 1 else target_name[0]

class BaseConcatDataset(ConcatDataset):
    """A base class for concatenated datasets. Holds either mne.Raw or
    mne.Epoch in self.datasets and has a pandas DataFrame with additional
    description.

    Parameters
    ----------
    list_of_ds : list
        list of BaseDataset, BaseConcatDataset or WindowsDataset
    target_transform : callable | None
        Optional function to call on targets before returning them.
    """
    def __init__(self, list_of_ds, target_transform=None):
        # if we get a list of BaseConcatDataset, get all the individual datasets
        if list_of_ds and isinstance(list_of_ds[0], BaseConcatDataset):
            list_of_ds = [d for ds in list_of_ds for d in ds.datasets]
        super().__init__(list_of_ds)

        self.target_transform = target_transform

    def _get_sequence(self, indices):
        X, y = list(), list()
        for ind in indices:
            out_i = super().__getitem__(ind)
            X.append(out_i[0])
            y.append(out_i[1])

        X = np.stack(X, axis=0)
        y = np.array(y)

        return X, y

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int | list
            Index of window and target to return. If provided as a list of
            ints, multiple windows and targets will be extracted and
            concatenated. The target output can be modified on the
            fly by the ``traget_transform`` parameter.
        """
        if isinstance(idx, Iterable):  # Sample multiple windows
            item = self._get_sequence(idx)
        else:
            item = super().__getitem__(idx)
        if self.target_transform is not None:
            item = item[:1] + (self.target_transform(item[1]),) + item[2:]
        return item

    def split(self, by=None, property=None, split_ids=None):
        """Split the dataset based on information listed in its description
        DataFrame or based on indices.

        Parameters
        ----------
        by : str | list | dict
            If ``by`` is a string, splitting is performed based on the
            description DataFrame column with this name.
            If ``by`` is a (list of) list of integers, the position in the first
            list corresponds to the split id and the integers to the
            datapoints of that split.
            If a dict then each key will be used in the returned
            splits dict and each value should be a list of int.
        property : str
            Some property which is listed in info DataFrame.
        split_ids : list | dict
            List of indices to be combined in a subset.
            It can be a list of int or a list of list of int.

        Returns
        -------
        splits : dict
            A dictionary with the name of the split (a string) as key and the
            dataset as value.
        """
        args_not_none = [
            by is not None, property is not None, split_ids is not None]
        if sum(args_not_none) != 1:
            raise ValueError("Splitting requires exactly one argument.")

        if property is not None or split_ids is not None:
            warnings.warn("Keyword arguments `property` and `split_ids` "
                          "are deprecated and will be removed in the future. "
                          "Use `by` instead.", DeprecationWarning)
            by = property if property is not None else split_ids
        if isinstance(by, str):
            split_ids = {
                k: list(v)
                for k, v in self.description.groupby(by).groups.items()
            }
        elif isinstance(by, dict):
            split_ids = by
        else:
            # assume list(int)
            if not isinstance(by[0], list):
                by = [by]
            # assume list(list(int))
            split_ids = {split_i: split for split_i, split in enumerate(by)}

        return {str(split_name): BaseConcatDataset(
            [self.datasets[ds_ind] for ds_ind in ds_inds], target_transform=self.target_transform)
            for split_name, ds_inds in split_ids.items()}

    def get_metadata(self):
        """Concatenate the metadata and description of the wrapped Epochs.

        Returns
        -------
        metadata : pd.DataFrame
            DataFrame containing as many rows as there are windows in the
            BaseConcatDataset, with the metadata and description information
            for each window.
        """
        if not all([isinstance(ds, (WindowsDataset, EEGWindowsDataset)) for ds in self.datasets]):
            raise TypeError('Metadata dataframe can only be computed when all '
                            'datasets are WindowsDataset.')

        all_dfs = list()
        for ds in self.datasets:
            if hasattr(ds, 'windows'):
                df = ds.windows.metadata
            else:
                df = ds.metadata
            for k, v in ds.description.items():
                df[k] = v
            all_dfs.append(df)

        return pd.concat(all_dfs)

    @property
    def transform(self):
        return [ds.transform for ds in self.datasets]

    @transform.setter
    def transform(self, fn):
        for i in range(len(self.datasets)):
            self.datasets[i].transform = fn

    @property
    def target_transform(self):
        return self._target_transform

    @target_transform.setter
    def target_transform(self, fn):
        if not (callable(fn) or fn is None):
            raise TypeError('target_transform must be a callable.')
        self._target_transform = fn

    def _outdated_save(self, path, overwrite=False):
        """This is a copy of the old saving function, that had inconsistent
        functionality for BaseDataset and WindowsDataset. It only exists to
        assure backwards compatibility by still being able to run the old tests.

        Save dataset to files.

        Parameters
        ----------
        path : str
            Directory to which .fif / -epo.fif and .json files are stored.
        overwrite : bool
            Whether to delete old files (.json, .fif, -epo.fif) in specified
            directory prior to saving.
        """
        warnings.warn('This function only exists for backwards compatibility '
                      'purposes. DO NOT USE!', UserWarning)
        if isinstance(self.datasets[0], EEGWindowsDataset):
            raise NotImplementedError("Outdated save not implemented for new window datasets.")
        if len(self.datasets) == 0:
            raise ValueError("Expect at least one dataset")
        if not (hasattr(self.datasets[0], 'raw') or hasattr(
                self.datasets[0], 'windows')):
            raise ValueError("dataset should have either raw or windows "
                             "attribute")
        file_name_templates = ["{}-raw.fif", "{}-epo.fif"]
        description_file_name = os.path.join(path, 'description.json')
        target_file_name = os.path.join(path, 'target_name.json')
        if not overwrite:
            from braindecode.datautil.serialization import \
                _check_save_dir_empty  # Import here to avoid circular import
            _check_save_dir_empty(path)
        else:
            for file_name_template in file_name_templates:
                file_names = glob(os.path.join(
                    path, f"*{file_name_template.lstrip('{}')}"))
                _ = [os.remove(f) for f in file_names]
            if os.path.isfile(target_file_name):
                os.remove(target_file_name)
            if os.path.isfile(description_file_name):
                os.remove(description_file_name)
            for kwarg_name in ['raw_preproc_kwargs', 'window_kwargs',
                               'window_preproc_kwargs']:
                kwarg_path = os.path.join(path, '.'.join([kwarg_name, 'json']))
                if os.path.exists(kwarg_path):
                    os.remove(kwarg_path)

        is_raw = hasattr(self.datasets[0], 'raw')

        if is_raw:
            file_name_template = file_name_templates[0]
        else:
            file_name_template = file_name_templates[1]

        for i_ds, ds in enumerate(self.datasets):
            full_file_path = os.path.join(path, file_name_template.format(i_ds))
            if is_raw:
                ds.raw.save(full_file_path, overwrite=overwrite)
            else:
                ds.windows.save(full_file_path, overwrite=overwrite)

        self.description.to_json(description_file_name)
        for kwarg_name in ['raw_preproc_kwargs', 'window_kwargs',
                           'window_preproc_kwargs']:
            if hasattr(self, kwarg_name):
                kwargs_path = os.path.join(path, '.'.join([kwarg_name, 'json']))
                kwargs = getattr(self, kwarg_name)
                if kwargs is not None:
                    json.dump(kwargs, open(kwargs_path, 'w'))

    @property
    def description(self):
        df = pd.DataFrame([ds.description for ds in self.datasets])
        df.reset_index(inplace=True, drop=True)
        return df

    def set_description(self, description, overwrite=False):
        """Update (add or overwrite) the dataset description.

        Parameters
        ----------
        description: dict | pd.DataFrame
            Description in the form key: value where the length of the value
            has to match the number of datasets.
        overwrite: bool
            Has to be True if a key in description already exists in the
            dataset description.
        """
        description = pd.DataFrame(description)
        for key, value in description.items():
            for ds, value_ in zip(self.datasets, value):
                ds.set_description({key: value_}, overwrite=overwrite)

    def save(self, path, overwrite=False, offset=0):
        """Save datasets to files by creating one subdirectory for each dataset:
        path/
            0/
                0-raw.fif | 0-epo.fif
                description.json
                raw_preproc_kwargs.json (if raws were preprocessed)
                window_kwargs.json (if this is a windowed dataset)
                window_preproc_kwargs.json  (if windows were preprocessed)
                target_name.json (if target_name is not None and dataset is raw)
            1/
                1-raw.fif | 1-epo.fif
                description.json
                raw_preproc_kwargs.json (if raws were preprocessed)
                window_kwargs.json (if this is a windowed dataset)
                window_preproc_kwargs.json  (if windows were preprocessed)
                target_name.json (if target_name is not None and dataset is raw)

        Parameters
        ----------
        path : str
            Directory in which subdirectories are created to store
             -raw.fif | -epo.fif and .json files to.
        overwrite : bool
            Whether to delete old subdirectories that will be saved to in this
            call.
        offset : int
            If provided, the integer is added to the id of the dataset in the
            concat. This is useful in the setting of very large datasets, where
            one dataset has to be processed and saved at a time to account for
            its original position.
        """
        if len(self.datasets) == 0:
            raise ValueError("Expect at least one dataset")
        if not (hasattr(self.datasets[0], 'raw') or hasattr(
                self.datasets[0], 'windows')):
            raise ValueError("dataset should have either raw or windows "
                             "attribute")
        path_contents = os.listdir(path)
        n_sub_dirs = len([os.path.isdir(e) for e in path_contents])
        for i_ds, ds in enumerate(self.datasets):
            # remove subdirectory from list of untouched files / subdirectories
            if str(i_ds + offset) in path_contents:
                path_contents.remove(str(i_ds + offset))
            # save_dir/i_ds/
            sub_dir = os.path.join(path, str(i_ds + offset))
            if os.path.exists(sub_dir):
                if overwrite:
                    shutil.rmtree(sub_dir)
                else:
                    raise FileExistsError(
                        f'Subdirectory {sub_dir} already exists. Please select'
                        f' a different directory, set overwrite=True, or '
                        f'resolve manually.')
            # save_dir/{i_ds+offset}/
            os.makedirs(sub_dir)
            # save_dir/{i_ds+offset}/{i_ds+offset}-{raw_or_epo}.fif
            self._save_signals(sub_dir, ds, i_ds, offset)
            # save_dir/{i_ds+offset}/metadata_df.pkl
            self._save_metadata(sub_dir, ds)
            # save_dir/{i_ds+offset}/description.json
            self._save_description(sub_dir, ds.description)
            # save_dir/{i_ds+offset}/raw_preproc_kwargs.json
            # save_dir/{i_ds+offset}/window_kwargs.json
            # save_dir/{i_ds+offset}/window_preproc_kwargs.json
            self._save_kwargs(sub_dir, ds)
            # save_dir/{i_ds+offset}/target_name.json
            self._save_target_name(sub_dir, ds)
        if overwrite:
            # the following will be True for all datasets preprocessed and
            # stored in parallel with braindecode.preprocessing.preprocess
            if i_ds+1+offset < n_sub_dirs:
                warnings.warn(f"The number of saved datasets ({i_ds+1+offset}) "
                              f"does not match the number of existing "
                              f"subdirectories ({n_sub_dirs}). You may now "
                              f"encounter a mix of differently preprocessed "
                              f"datasets!", UserWarning)
        # if path contains files or directories that were not touched, raise
        # warning
        if path_contents:
            warnings.warn(f'Chosen directory {path} contains other '
                          f'subdirectories or files {path_contents}.')

    @staticmethod
    def _save_signals(sub_dir, ds, i_ds, offset):
        raw_or_epo = 'raw' if hasattr(ds, 'raw') else 'epo'
        fif_file_name = f'{i_ds + offset}-{raw_or_epo}.fif'
        fif_file_path = os.path.join(sub_dir, fif_file_name)
        raw_or_windows = 'raw' if raw_or_epo == 'raw' else 'windows'

        # The following appears to be necessary to avoid a CI failure when
        # preprocessing WindowsDatasets with serialization enabled. The failure
        # comes from `mne.epochs._check_consistency` which ensures the Epochs's
        # object `times` attribute is not writeable.
        getattr(ds, raw_or_windows).times.flags['WRITEABLE'] = False

        getattr(ds, raw_or_windows).save(fif_file_path)

    @staticmethod
    def _save_metadata(sub_dir, ds):
        if hasattr(ds, 'metadata'):
            metadata_file_path = os.path.join(sub_dir, 'metadata_df.pkl')
            ds.metadata.to_pickle(metadata_file_path)

    @staticmethod
    def _save_description(sub_dir, description):
        description_file_path = os.path.join(sub_dir, 'description.json')
        description.to_json(description_file_path)

    @staticmethod
    def _save_kwargs(sub_dir, ds):
        for kwargs_name in ['raw_preproc_kwargs', 'window_kwargs',
                            'window_preproc_kwargs']:
            if hasattr(ds, kwargs_name):
                kwargs_file_name = '.'.join([kwargs_name, 'json'])
                kwargs_file_path = os.path.join(sub_dir, kwargs_file_name)
                kwargs = getattr(ds, kwargs_name)
                if kwargs is not None:
                    with open(kwargs_file_path, 'w') as f:
                        json.dump(kwargs, f)

    @staticmethod
    def _save_target_name(sub_dir, ds):
        if hasattr(ds, 'target_name'):
            target_file_path = os.path.join(sub_dir, 'target_name.json')
            with open(target_file_path, 'w') as f:
                json.dump({'target_name': ds.target_name}, f)

class EEGWindowsDataset(BaseDataset):
    """Returns windows from an mne.Raw object, its window indices, along with a target.

    Dataset which serves windows from an mne.Epochs object along with their
    target and additional information. The `metadata` attribute of the Epochs
    object must contain a column called `target`, which will be used to return
    the target that corresponds to a window. Additional columns
    `i_window_in_trial`, `i_start_in_trial`, `i_stop_in_trial` are also
    required to serve information about the windowing (e.g., useful for cropped
    training).
    See `braindecode.datautil.windowers` to directly create a `WindowsDataset`
    from a `BaseDataset` object.

    Parameters
    ----------
    windows : mne.Raw or mne.Epochs (Epochs is outdated)
        Windows obtained through the application of a windower to a BaseDataset
        (see `braindecode.datautil.windowers`).
    description : dict | pandas.Series | None
        Holds additional info about the windows.
    transform : callable | None
        On-the-fly transform applied to a window before it is returned.
    targets_from : str
        Defines whether targets will be extracted from  metadata or from `misc`
        channels (time series targets). It can be `metadata` (default) or `channels`.
    last_target_only : bool
        If targets are obtained from misc channels whether all targets if the entire
        (compute) window will be returned or only the last target in the window.
    metadata : pandas.DataFrame
        Dataframe with crop indices, so `i_window_in_trial`, `i_start_in_trial`, `i_stop_in_trial`
        as well as `targets`.
    """

    def __init__(self, raw, metadata, description=None, transform=None, targets_from='metadata',
                 last_target_only=True, ):
        self.raw = raw
        self.metadata = metadata
        self._description = _create_description(description)

        self.transform = transform
        self.last_target_only = last_target_only
        if targets_from not in ('metadata', 'channels'):
            raise ValueError('Wrong value for parameter `targets_from`.')
        self.targets_from = targets_from
        self.crop_inds = metadata.loc[
                         :, ['i_window_in_trial', 'i_start_in_trial',
                             'i_stop_in_trial']].to_numpy()
        if self.targets_from == 'metadata':
            self.y = metadata.loc[:, 'target'].to_list()

    def __getitem__(self, index):
        """Get a window and its target.

        Parameters
        ----------
        index : int
            Index to the window (and target) to return.

        Returns
        -------
        np.ndarray
            Window of shape (n_channels, n_times).
        int
            Target for the windows.
        np.ndarray
            Crop indices.
        """

        # necessary to cast as list to get list of three tensors from batch,
        # otherwise get single 2d-tensor...
        crop_inds = self.crop_inds[index].tolist()

        i_window_in_trial, i_start, i_stop = crop_inds
        X = self.raw._getitem((slice(None), slice(i_start, i_stop)), return_times=False)
        X = X.astype('float32')
        # ensure we don't give the user the option
        # to accidentally modify the underlying array
        X = X.copy()
        if self.transform is not None:
            X = self.transform(X)
        if self.targets_from == 'metadata':
            y = self.y[index]
        else:
            misc_mask = np.array(self.raw.get_channel_types()) == 'misc'
            if self.last_target_only:
                y = X[misc_mask, -1]
            else:
                y = X[misc_mask, :]
            # ensure we don't give the user the option
            # to accidentally modify the underlying array
            y = y.copy()
            # remove the target channels from raw
            X = X[~misc_mask, :]
        return X, y, crop_inds

    def __len__(self):
        return len(self.crop_inds)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        if value is not None and not callable(value):
            raise ValueError('Transform needs to be a callable.')
        self._transform = value

    @property
    def description(self):
        return self._description

    def set_description(self, description, overwrite=False):
        """Update (add or overwrite) the dataset description.

        Parameters
        ----------
        description: dict | pd.Series
            Description in the form key: value.
        overwrite: bool
            Has to be True if a key in description already exists in the
            dataset description.
        """
        description = _create_description(description)
        for key, value in description.items():
            # if they key is already in the existing description, drop it
            if key in self._description:
                assert overwrite, (f"'{key}' already in description. Please "
                                   f"rename or set overwrite to True.")
                self._description.pop(key)
        self._description = pd.concat([self.description, description])

class WindowsDataset(BaseDataset):
    """Returns windows from an mne.Epochs object along with a target.

    Dataset which serves windows from an mne.Epochs object along with their
    target and additional information. The `metadata` attribute of the Epochs
    object must contain a column called `target`, which will be used to return
    the target that corresponds to a window. Additional columns
    `i_window_in_trial`, `i_start_in_trial`, `i_stop_in_trial` are also
    required to serve information about the windowing (e.g., useful for cropped
    training).
    See `braindecode.datautil.windowers` to directly create a `WindowsDataset`
    from a `BaseDataset` object.

    Parameters
    ----------
    windows : mne.Epochs
        Windows obtained through the application of a windower to a BaseDataset
        (see `braindecode.datautil.windowers`).
    description : dict | pandas.Series | None
        Holds additional info about the windows.
    transform : callable | None
        On-the-fly transform applied to a window before it is returned.
    targets_from : str
        Defines whether targets will be extracted from mne.Epochs metadata or mne.Epochs `misc`
        channels (time series targets). It can be `metadata` (default) or `channels`.
    """
    def __init__(self, windows, description=None, transform=None, targets_from='metadata',
                 last_target_only=True):
        self.windows = windows
        self._description = _create_description(description)
        self.transform = transform
        self.last_target_only = last_target_only
        if targets_from not in ('metadata', 'channels'):
            raise ValueError('Wrong value for parameter `targets_from`.')
        self.targets_from = targets_from

        self.crop_inds = self.windows.metadata.loc[
            :, ['i_window_in_trial', 'i_start_in_trial',
                'i_stop_in_trial']].to_numpy()
        if self.targets_from == 'metadata':
            self.y = self.windows.metadata.loc[:, 'target'].to_list()

    def __getitem__(self, index):
        """Get a window and its target.

        Parameters
        ----------
        index : int
            Index to the window (and target) to return.

        Returns
        -------
        np.ndarray
            Window of shape (n_channels, n_times).
        int
            Target for the windows.
        np.ndarray
            Crop indices.
        """
        X = self.windows.get_data(item=index)[0].astype('float32')
        if self.transform is not None:
            X = self.transform(X)
        if self.targets_from == 'metadata':
            y = self.y[index]
        else:
            misc_mask = np.array(self.windows.get_channel_types()) == 'misc'
            if self.last_target_only:
                y = X[misc_mask, -1]
            else:
                y = X[misc_mask, :]
            # remove the target channels from raw
            X = X[~misc_mask, :]
        # necessary to cast as list to get list of three tensors from batch,
        # otherwise get single 2d-tensor...
        crop_inds = self.crop_inds[index].tolist()
        return X, y, crop_inds

    def __len__(self):
        return len(self.windows.events)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        if value is not None and not callable(value):
            raise ValueError('Transform needs to be a callable.')
        self._transform = value

    @property
    def description(self):
        return self._description

    def set_description(self, description, overwrite=False):
        """Update (add or overwrite) the dataset description.

        Parameters
        ----------
        description: dict | pd.Series
            Description in the form key: value.
        overwrite: bool
            Has to be True if a key in description already exists in the
            dataset description.
        """
        description = _create_description(description)
        for key, value in description.items():
            # if they key is already in the existing description, drop it
            if key in self._description:
                assert overwrite, (f"'{key}' already in description. Please "
                                   f"rename or set overwrite to True.")
                self._description.pop(key)
        self._description = pd.concat([self.description, description])

class TUH(BaseConcatDataset):
    """Temple University Hospital (TUH) EEG Corpus
    (www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tueg).

    Parameters
    ----------
    path: str
        Parent directory of the dataset.
    recording_ids: list(int) | int
        A (list of) int of recording id(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later then the second recording. Provide recording_ids in ascending
        order to preserve chronological order.).
    target_name: str
        Can be 'gender', or 'age'.
    preload: bool
        If True, preload the data of the Raw objects.
    add_physician_reports: bool
        If True, the physician reports will be read from disk and added to the
        description.
    rename_channels: bool
        If True, rename the EEG channels to the standard 10-05 system.
    set_montage: bool
        If True, set the montage to the standard 10-05 system.
    n_jobs: int
        Number of jobs to be used to read files in parallel.
    """

    def __init__(
        self,
        path: str,
        recording_ids: list[int] | None = None,
        target_name: str | tuple[str, ...] | None = None,
        preload: bool = False,
        add_physician_reports: bool = False,
        rename_channels: bool = False,
        set_montage: bool = False,
        n_jobs: int = 1,
    ):
        if set_montage:
            assert (
                rename_channels
            ), "If set_montage is True, rename_channels must be True."
        # create an index of all files and gather easily accessible info
        # without actually touching the files
        file_paths = glob.glob(os.path.join(path, "**/*.edf"), recursive=True)
        #keep only files that end with .edf
        file_paths = [f for f in file_paths if f.endswith(".edf")]
        descriptions = _create_description(file_paths)
        # sort the descriptions chronologicaly
        descriptions = _sort_chronologically(descriptions)
        # limit to specified recording ids before doing slow stuff
        if recording_ids is not None:
            if not isinstance(recording_ids, Iterable):
                # Assume it is an integer specifying number of recordings to load
                recording_ids = range(recording_ids)
            descriptions = descriptions[recording_ids]

    
        if n_jobs == 1:
            base_datasets = [
                self._create_dataset(
                    descriptions[i],
                    target_name,
                    preload,
                    add_physician_reports,
                    rename_channels,
                    set_montage,
                )
                for i in descriptions.columns
            ]
        else:
            base_datasets = Parallel(n_jobs)(
                delayed(self._create_dataset)(
                    descriptions[i],
                    target_name,
                    preload,
                    add_physician_reports,
                    rename_channels,
                    set_montage,
                )
                for i in descriptions.columns
            )
        super().__init__(base_datasets)

    @staticmethod
    def _rename_channels(raw):
        """

        
        Renames the EEG channels in the given mne.io.Raw object according to MNE conventions and sets their channel types accordingly.

        This function:
            - Strips reference suffixes and prefixes from channel names (e.g., "-REF", "-LE", "EEG ").
            - Maps recognized channels to a standard 10-20 montage and sets them to type 'eeg'.
            - Marks channels that cannot be inferred (defaulted to 'eeg') as 'misc'.
            - Preserves channel name casing by renaming channels to their correct 10-20 format if available.

        See also
        --------
        https://isip.piconepress.com/publications/reports/2020/tuh_eeg/electrodes/

        Parameters
        ----------
        raw : mne.io.Raw
                The MNE Raw object containing EEG data to be processed.

        Returns
        -------
        None
        Renames the EEG channels using mne conventions and sets their type to 'eeg'.

        See https://isip.piconepress.com/publications/reports/2020/tuh_eeg/electrodes/
        """
        # remove ref suffix and prefix:
        # TODO: replace with removesuffix and removeprefix when 3.8 is dropped
        mapping_strip = {
            c: c.replace("-REF", "").replace("-LE", "").replace("EEG ", "")
            for c in raw.ch_names
        }
        raw.rename_channels(mapping_strip)

        montage1020 = mne.channels.make_standard_montage("standard_1020")
        mapping_eeg_names = {
            c.upper(): c for c in montage1020.ch_names if c.upper() in raw.ch_names
        }

        # Set channels whose type could not be inferred (defaulted to "eeg") to "misc":
        non_eeg_names = [c for c in raw.ch_names if c not in mapping_eeg_names]
        if non_eeg_names:
            non_eeg_types = raw.get_channel_types(picks=non_eeg_names)
            mapping_non_eeg_types = {
                c: "misc" for c, t in zip(non_eeg_names, non_eeg_types) if t == "eeg"
            }
            if mapping_non_eeg_types:
                raw.set_channel_types(mapping_non_eeg_types, verbose="error")

        if mapping_eeg_names:
            # Set 1005 channels type to "eeg":
            raw.set_channel_types(
                {c: "eeg" for c in mapping_eeg_names}, on_unit_change="ignore"
            )
            # Fix capitalized EEG channel names:
            raw.rename_channels(mapping_eeg_names)

    @staticmethod
    def _set_montage(raw):
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="warn")

    @staticmethod
    def _create_dataset(
        description,
        target_name,
        preload,
        add_physician_reports,
        rename_channels,
        set_montage,
    ):
        file_path = description.loc["path"]

        # parse age and gender information from EDF header
        age, gender = _parse_age_and_gender_from_edf_header(file_path)
        raw = mne.io.read_raw_edf(
            file_path, preload=preload, infer_types=True, verbose="error"
        )
        if rename_channels:
            TUH._rename_channels(raw)
        if set_montage:
            TUH._set_montage(raw)

        meas_date = (
            datetime(1, 1, 1, tzinfo=timezone.utc)
            if raw.info["meas_date"] is None
            else raw.info["meas_date"]
        )
        # if this is old version of the data and the year could be parsed from
        # file paths, use this instead as before
        if "year" in description:
            meas_date = meas_date.replace(*description[["year", "month", "day"]])
        raw.set_meas_date(meas_date)

        d = {
            "age": int(age),
            "gender": gender,
        }
        # if year exists in description = old version
        # if not, get it from meas_date in raw.info and add to description
        # if meas_date is None, create fake one
        if "year" not in description:
            d["year"] = raw.info["meas_date"].year
            d["month"] = raw.info["meas_date"].month
            d["day"] = raw.info["meas_date"].day

        # read info relevant for preprocessing from raw without loading it
        if add_physician_reports:
            physician_report = _read_physician_report(file_path)
            d["report"] = physician_report
        additional_description = pd.Series(d)
        description = pd.concat([description, additional_description])
        base_dataset = BaseDataset(raw, description, target_name=target_name)
        return base_dataset

class TUHAbnormal(TUH):
    """Temple University Hospital (TUH) Abnormal EEG Corpus.
    see www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tuab

    Parameters
    ----------
    path: str
        Parent directory of the dataset.
    recording_ids: list(int) | int
        A (list of) int of recording id(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later then the second recording. Provide recording_ids in ascending
        order to preserve chronological order.).
    target_name: str
        Can be 'pathological', 'gender', or 'age'.
    preload: bool
        If True, preload the data of the Raw objects.
    add_physician_reports: bool
        If True, the physician reports will be read from disk and added to the
        description.
    rename_channels: bool
        If True, rename the EEG channels to the standard 10-05 system.
    set_montage: bool
        If True, set the montage to the standard 10-05 system.
    n_jobs: int
        Number of jobs to be used to read files in parallel.
    """

    def __init__(
        self,
        path: str,
        recording_ids: list[int] | None = None,
        target_name: str | tuple[str, ...] | None = "pathological",
        preload: bool = False,
        add_physician_reports: bool = False,
        rename_channels: bool = False,
        set_montage: bool = False,
        n_jobs: int = 1,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*not in description. '__getitem__'"
            )
            super().__init__(
                path=path,
                recording_ids=recording_ids,
                preload=preload,
                target_name=target_name,
                add_physician_reports=add_physician_reports,
                rename_channels=rename_channels,
                set_montage=set_montage,
                n_jobs=n_jobs,
            )
        additional_descriptions = []
        for file_path in self.description.path:
            additional_description = self._parse_additional_description_from_file_path(
                file_path
            )
            additional_descriptions.append(additional_description)
        additional_descriptions = pd.DataFrame(additional_descriptions)
        self.set_description(additional_descriptions, overwrite=True)

    @staticmethod
    def _parse_additional_description_from_file_path(file_path):
        file_path = os.path.normpath(file_path)
        tokens = file_path.split(os.sep)
        # expect paths as version/file type/data_split/pathology status/
        #                     reference/subset/subject/recording session/file
        # e.g.            v2.0.0/edf/train/normal/01_tcp_ar/000/00000021/
        #                     s004_2013_08_15/00000021_s004_t000.edf
        assert "abnormal" in tokens or "normal" in tokens, "No pathology labels found."
        assert (
            "train" in tokens or "eval" in tokens
        ), "No train or eval set information found."
        return {
            "version": tokens[-9],
            "train": "train" in tokens,
            "pathological": "abnormal" in tokens,
        }

class TUHEpilepsy(TUH):
    def __init__(
        self,
        path: str,
        recording_ids: list[int] | None = None,
        target_name: str | tuple[str, ...] | None = None, # Can be 
        preload: bool = False,
        add_physician_reports: bool = False,
        rename_channels: bool = False,
        set_montage: bool = False,
        n_jobs: int = 1,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*not in description. '__getitem__'"
            )
            super().__init__(
                path=path,
                recording_ids=recording_ids,
                preload=preload,
                target_name=target_name,
                add_physician_reports=add_physician_reports,
                rename_channels=rename_channels,
                set_montage=set_montage,
                n_jobs=n_jobs,
            )
        additional_descriptions = []

        for file_path in self.description["path"]:
            additional_description = self._parse_additional_description_from_file_path(
                file_path
            )
            additional_descriptions.append(additional_description)
        additional_descriptions = pd.DataFrame(additional_descriptions)
        self.set_description(additional_descriptions, overwrite=True)

    @staticmethod
    def _parse_additional_description_from_file_path(file_path):
        file_path = os.path.normpath(file_path)
        tokens = file_path.split(os.sep)

        if '00_epilepsy' in tokens:
            return {
                "epilepsy": 0
            }
        elif '01_no_epilepsy' in tokens:
            return {
                "epilepsy": 1
            }
        else:
            raise ValueError("No epilepsy labels found.")
        
    

def _create_description(description):
    if description is None:
        return None
    if isinstance(description, pd.Series):
        return description
    if isinstance(description, dict):
        return pd.Series(description)
    if isinstance(description, pd.DataFrame):
        return description

    if isinstance(description, list):
        if not description:
            return pd.DataFrame()
        first = description[0]
        if isinstance(first, (dict, pd.Series)):
            return pd.DataFrame(description).T
        descriptions = [_parse_description_from_file_path(p) for p in description]
        return pd.DataFrame(descriptions).T

    if isinstance(description, str):
        return pd.DataFrame([_parse_description_from_file_path(description)]).T

    raise TypeError("Unsupported description type")

def _sort_chronologically(descriptions):
    descriptions.sort_values(
        ["year", "month", "day", "subject", "session", "segment"], axis=1, inplace=True
    )
    return descriptions

def _read_date(file_path):
    date_path = file_path.replace(".edf", "_date.txt")
    # if date file exists, read it
    if os.path.exists(date_path):
        description = pd.read_json(date_path, typ="series").to_dict()
    # otherwise read edf file, extract date and store to file
    else:
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose="error")
        description = {
            "year": raw.info["meas_date"].year,
            "month": raw.info["meas_date"].month,
            "day": raw.info["meas_date"].day,
        }
        # if the txt file storing the recording date does not exist, create it
        try:
            pd.Series(description).to_json(date_path)
        except OSError:
            warnings.warn(
                f"Cannot save date file to {date_path}. "
                f"This might slow down creation of the dataset."
            )
    return description

def _parse_description_from_file_path(file_path):
    # stackoverflow.com/questions/3167154/how-to-split-a-dos-path-into-its-components-in-python  # noqa

    file_path = os.path.normpath(file_path)
    tokens = file_path.split(os.sep)
    version = tokens[-6]

    #subject_id = tokens[-1].split("_")[0] # OG backup
    subject_id = tokens[-4]

    #session = tokens[-2].split("_")[0]  # OG BACKUP string on format 's000'
    session= tokens[-3].split("_")[0]
    
    # According to the example path in the comment 8 lines above segment is not included in the file name
    segment = tokens[-1].split("_")[-1].split(".")[0]

    year= tokens[-3].split("_")[1:][0]

    return {
            "path": file_path,
            "version": version,
            "year": int(year),
            "month": int(1),
            "day": int(1),
            "subject": subject_id,
            "session": int(session[1:]),
            "segment": int(segment[1:]),
        }

def _read_physician_report(file_path):
    directory = os.path.dirname(file_path)
    txt_file = glob.glob(os.path.join(directory, "**/*.txt"), recursive=True)
    # check that there is at most one txt file in the same directory
    assert len(txt_file) in [0, 1]
    report = ""
    if txt_file:
        txt_file = txt_file[0]
        # somewhere in the corpus, encoding apparently changed
        # first try to read as utf-8, if it does not work use latin-1
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                report = f.read()
        except UnicodeDecodeError:
            with open(txt_file, "r", encoding="latin-1") as f:
                report = f.read()
    if not report:
        raise RuntimeError(
            f"Could not read physician report ({txt_file}). "
            f"Disable option or choose appropriate directory."
        )
    return report

def _read_edf_header(file_path):
    f = open(file_path, "rb")
    header = f.read(88)
    f.close()
    return header

def _parse_age_and_gender_from_edf_header(file_path):
    header = _read_edf_header(file_path)
    # bytes 8 to 88 contain ascii local patient identification
    # see https://www.teuniz.net/edfbrowser/edf%20format%20description.html
    patient_id = header[8:].decode("ascii")
    age = -1
    found_age = re.findall(r"Age:(\d+)", patient_id)
    if len(found_age) == 1:
        age = int(found_age[0])
    gender = "X"
    found_gender = re.findall(r"\s([F|M])\s", patient_id)
    if len(found_gender) == 1:
        gender = found_gender[0]
    return age, gender

def _fake_raw(*args, **kwargs):
    sfreq = 10
    ch_names = [
        "EEG A1-REF",
        "EEG A2-REF",
        "EEG FP1-REF",
        "EEG FP2-REF",
        "EEG F3-REF",
        "EEG F4-REF",
        "EEG C3-REF",
        "EEG C4-REF",
        "EEG P3-REF",
        "EEG P4-REF",
        "EEG O1-REF",
        "EEG O2-REF",
        "EEG F7-REF",
        "EEG F8-REF",
        "EEG T3-REF",
        "EEG T4-REF",
        "EEG T5-REF",
        "EEG T6-REF",
        "EEG FZ-REF",
        "EEG CZ-REF",
        "EEG PZ-REF",
    ]
    duration_min = 6
    data = np.random.randn(len(ch_names), duration_min * sfreq * 60)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data=data, info=info)
    return raw

def _get_header(*args, **kwargs):
    all_paths = {**_TUH_EEG_PATHS, **_TUH_EEG_ABNORMAL_PATHS}
    return all_paths[args[0]]

_TUH_EEG_PATHS = {
    # These are actual file paths and edf headers from the TUH EEG Corpus (v1.1.0 and v1.2.0)
    "tuh_eeg/v1.1.0/edf/01_tcp_ar/000/00000000/s001_2015_12_30/00000000_s001_t000.edf": b"0       00000000 M 01-JAN-1978 00000000 Age:37                                          ",
    # noqa E501
    "tuh_eeg/v1.1.0/edf/01_tcp_ar/099/00009932/s004_2014_09_30/00009932_s004_t013.edf": b"0       00009932 F 01-JAN-1961 00009932 Age:53                                          ",
    # noqa E501
    "tuh_eeg/v1.1.0/edf/02_tcp_le/000/00000058/s001_2003_02_05/00000058_s001_t000.edf": b"0       00000058 M 01-JAN-2003 00000058 Age:0.0109                                      ",
    # noqa E501
    "tuh_eeg/v1.1.0/edf/03_tcp_ar_a/123/00012331/s003_2014_12_14/00012331_s003_t002.edf": b"0       00012331 M 01-JAN-1975 00012331 Age:39                                          ",
    # noqa E501
    "tuh_eeg/v1.2.0/edf/03_tcp_ar_a/149/00014928/s004_2016_01_15/00014928_s004_t007.edf": b"0       00014928 F 01-JAN-1933 00014928 Age:83                                          ",
    # noqa E501
}
_TUH_EEG_ABNORMAL_PATHS = {
    # these are actual file paths and edf headers from TUH Abnormal EEG Corpus (v2.0.0)
    "tuh_abnormal_eeg/v2.0.0/edf/train/normal/01_tcp_ar/078/00007871/s001_2011_07_05/00007871_s001_t001.edf": b"0       00007871 F 01-JAN-1988 00007871 Age:23                                          ",
    # noqa E501
    "tuh_abnormal_eeg/v2.0.0/edf/train/normal/01_tcp_ar/097/00009777/s001_2012_09_17/00009777_s001_t000.edf": b"0       00009777 M 01-JAN-1986 00009777 Age:26                                          ",
    # noqa E501
    "tuh_abnormal_eeg/v2.0.0/edf/train/abnormal/01_tcp_ar/083/00008393/s002_2012_02_21/00008393_s002_t000.edf": b"0       00008393 M 01-JAN-1960 00008393 Age:52                                          ",
    # noqa E501
    "tuh_abnormal_eeg/v2.0.0/edf/train/abnormal/01_tcp_ar/012/00001200/s003_2010_12_06/00001200_s003_t000.edf": b"0       00001200 M 01-JAN-1963 00001200 Age:47                                          ",
    # noqa E501
    "tuh_abnormal_eeg/v2.0.0/edf/eval/abnormal/01_tcp_ar/059/00005932/s004_2013_03_14/00005932_s004_t000.edf": b"0       00005932 M 01-JAN-1963 00005932 Age:50                                          ",
    # noqa E501
}

class _TUHMock(TUH):
    """Mocked class for testing and examples."""

    @mock.patch("glob.glob", return_value=_TUH_EEG_PATHS.keys())
    @mock.patch("mne.io.read_raw_edf", new=_fake_raw)
    @mock.patch("braindecode.datasets.tuh._read_edf_header", new=_get_header)
    def __init__(
        self,
        mock_glob,
        path: str,
        recording_ids: list[int] | None = None,
        target_name: str | tuple[str, ...] | None = None,
        preload: bool = False,
        add_physician_reports: bool = False,
        rename_channels: bool = False,
        set_montage: bool = False,
        n_jobs: int = 1,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Cannot save date file")
            super().__init__(
                path=path,
                recording_ids=recording_ids,
                target_name=target_name,
                preload=preload,
                add_physician_reports=add_physician_reports,
                rename_channels=rename_channels,
                set_montage=set_montage,
                n_jobs=n_jobs,
            )

class _TUHAbnormalMock(TUHAbnormal):
    """Mocked class for testing and examples."""

    @mock.patch("glob.glob", return_value=_TUH_EEG_ABNORMAL_PATHS.keys())
    @mock.patch("mne.io.read_raw_edf", new=_fake_raw)
    @mock.patch("braindecode.datasets.tuh._read_edf_header", new=_get_header)
    @mock.patch(
        "braindecode.datasets.tuh._read_physician_report", return_value="simple_test"
    )
    def __init__(
        self,
        mock_glob,
        mock_report,
        path: str,
        recording_ids: list[int] | None = None,
        target_name: str | tuple[str, ...] | None = "pathological",
        preload: bool = False,
        add_physician_reports: bool = False,
        rename_channels: bool = False,
        set_montage: bool = False,
        n_jobs: int = 1,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Cannot save date file")
            super().__init__(
                path=path,
                recording_ids=recording_ids,
                target_name=target_name,
                preload=preload,
                add_physician_reports=add_physician_reports,
                rename_channels=rename_channels,
                set_montage=set_montage,
                n_jobs=n_jobs,
            )            
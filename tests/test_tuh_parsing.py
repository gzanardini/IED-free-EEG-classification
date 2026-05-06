import pathlib
import sys

import numpy as np
import pandas as pd
import mne
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import TUEP as tuep


def _make_raw(n_samples=10, sfreq=10.0):
    data = np.zeros((1, n_samples))
    info = mne.create_info(["EEG FP1-REF"], sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="error")


def test_parse_description_from_file_path_custom_layout():
    path = (
        "/root/v1.1.0/01_tcp_ar/000/00000000/"
        "s001_2015_12_30/00000000_s001_t000.edf"
    )
    desc = tuep._parse_description_from_file_path(path)

    assert desc["version"] == "v1.1.0"
    assert desc["subject"] == "00000000"
    assert desc["session"] == 1
    assert desc["segment"] == 0
    assert desc["year"] == 2015


def test_parse_age_and_gender_from_edf_header_uses_mocked_header(monkeypatch):
    sample_path = next(iter(tuep._TUH_EEG_PATHS.keys()))
    header = tuep._TUH_EEG_PATHS[sample_path]

    monkeypatch.setattr(tuep, "_read_edf_header", lambda _: header)
    age, gender = tuep._parse_age_and_gender_from_edf_header(sample_path)

    assert age == 37
    assert gender == "M"


@pytest.mark.parametrize(
    "path,train_expected,pathological_expected",
    [
        (
            "tuh_abnormal_eeg/v2.0.0/edf/train/normal/01_tcp_ar/078/"
            "00007871/s001_2011_07_05/00007871_s001_t001.edf",
            True,
            False,
        ),
        (
            "tuh_abnormal_eeg/v2.0.0/edf/train/abnormal/01_tcp_ar/083/"
            "00008393/s002_2012_02_21/00008393_s002_t000.edf",
            True,
            True,
        ),
        (
            "tuh_abnormal_eeg/v2.0.0/edf/eval/abnormal/01_tcp_ar/059/"
            "00005932/s004_2013_03_14/00005932_s004_t000.edf",
            False,
            True,
        ),
    ],
)
def test_tuhabnormal_additional_description_flags(
    path, train_expected, pathological_expected
):
    desc = tuep.TUHAbnormal._parse_additional_description_from_file_path(path)

    assert desc["version"] == "v2.0.0"
    assert desc["train"] is train_expected
    assert desc["pathological"] is pathological_expected


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/root/00_epilepsy/subject/file.edf", 0),
        ("/root/01_no_epilepsy/subject/file.edf", 1),
    ],
)
def test_tuhepilepsy_additional_description_flags(path, expected):
    desc = tuep.TUHEpilepsy._parse_additional_description_from_file_path(path)

    assert desc["epilepsy"] == expected


def test_base_dataset_target_selection(monkeypatch):
    def _simple_description(obj):
        if isinstance(obj, pd.Series):
            return obj
        if isinstance(obj, dict):
            return pd.Series(obj)
        raise TypeError("Unsupported description type")

    monkeypatch.setattr(tuep, "_create_description", _simple_description)
    raw = _make_raw()

    ds_epilepsy = tuep.BaseDataset(
        raw, description={"epilepsy": 1, "age": 12}, target_name="epilepsy"
    )
    _, y_epilepsy = ds_epilepsy[0]

    ds_age = tuep.BaseDataset(raw, description={"epilepsy": 1, "age": 12}, target_name="age")
    _, y_age = ds_age[0]

    ds_multi = tuep.BaseDataset(
        raw, description={"age": 12, "gender": "M"}, target_name=("age", "gender")
    )
    _, y_multi = ds_multi[0]

    assert y_epilepsy == 1
    assert y_age == 12
    assert y_multi == [12, "M"]

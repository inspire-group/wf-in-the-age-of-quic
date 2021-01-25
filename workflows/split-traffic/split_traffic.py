#!/usr/bin/env python3
"""Usage: split_traffic [options] INFILE HDF_FILE

Options:
    --freq f
        Split the data into k folds [default: 50]

    --strategy s
        Repeat the k-folds n times [default: rr]

    --n-paths n
        Set aside frac fraction of the training set for validation [default: 2].

"""

import h5py
import logging
import doceasy
import numpy as np
import random

from typing import Iterator, IO
_LOGGER = logging.getLogger("be-split-traffic")

# bookkeeping: keep this seed somewhere
random.seed(1234)

def choose_path(prev_path, npaths, strategy):
    if strategy == "rr":
        return (prev_path + 1) % npaths
    # TODO: be able to manually tune weighted random
    if strategy == "wr":
        return random.choice(range(npaths))

def split(label, sizes, timestamps, strategy, freq, npaths):
    if len(sizes) <= npaths * freq:
        return [label], [sizes], [timestamps]
    new_labels = []
    new_sizes = []
    new_timestamps = []
    for n in range(npaths):
        new_sizes.append([])
        new_timestamps.append([])

    path = 0
    for i in range(0, len(sizes), freq):
        new_sizes[path] += list(sizes[i:i+freq])
        new_timestamps[path] += list(timestamps[i:i+freq])
        path = (path + 1) % npaths

    # Normalize all new timestamp paths to 0, remove empty paths
    new_timestamps = [
            [time - timestamp[0] for time in timestamp] for timestamp in new_timestamps
            if len(timestamp) > 0
        ]

    new_sizes = [sizes for sizes in new_sizes if len(sizes) > 0]
    for n in range(len(sizes)):
        new_labels.append(label)

    return new_labels, new_sizes, new_timestamps

LABEL_DTYPE = np.dtype([("class", "i1"), ('group', '<i8'), ("protocol", "S10"), ("region", "S7")])
def _write_labels(label_data, hdf_file: h5py.File):
    """Append the labels found in trace_data to the HDF file."""
    labels = np.array(label_data, dtype=LABEL_DTYPE)

    if "/labels" not in hdf_file.keys():
        hdf_file.create_dataset(
            "/labels", data=labels, maxshape=(None, ),
            compression="gzip")
    else:
        new_length = hdf_file["/labels"].len() + len(labels)
        hdf_file["/labels"].resize(new_length, axis=0)
        hdf_file["/labels"][-len(labels):] = labels

SIZE_DTYPE = np.dtype("i4")
def _write_traces(trace_data, hdf_file: h5py.File):
    """Append the traces found in trace_data to the HDF file."""
    for key, raw_type in zip(("sizes", "timestamps"), (SIZE_DTYPE, float)):
        data = list([np.array(x, dtype=raw_type) for x in trace_data[key]])

        if key not in hdf_file.keys():
            dtype = h5py.vlen_dtype(raw_type)
            hdf_file.create_dataset(
                key, data=data, dtype=dtype, maxshape=(None,),
                compression="gzip")
        else:
            dataset = hdf_file[key]
            new_length = dataset.len() + len(data)
            dataset.resize(new_length, axis=0)
            dataset[-len(data):] = data


def main(infile: str, hdf_file: str, **splitter_kw):
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s',
        level=logging.INFO)

    strategy = splitter_kw.pop("strategy")
    freq = splitter_kw.pop("freq")
    npaths = splitter_kw.pop("n_paths")

    with h5py.File(infile, mode="r") as h5in:
        _LOGGER.info("Reading labels from %r...", infile)
        #labels = (pd.DataFrame.from_records(np.asarray(h5in["labels"]))
        #          .transform(decode_column))
        labels = h5in["labels"]

        ls = []
        sizes = []
        timestamps = []
        for i in range(len(labels)):
            if labels[i][2] != b"quic":
                continue
            l, s, t = split(labels[i], h5in["sizes"][i], h5in["timestamps"][i], strategy, freq, npaths)
            ls += l
            sizes += s
            timestamps += t
        with h5py.File(hdf_file, mode="w") as outfile:
            _write_labels(ls, outfile)
            _write_traces({"sizes": sizes, "timestamps": timestamps}, outfile)


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, {
        "INFILE": str,
        "HDF_FILE": str,
        "--freq": doceasy.Use(int),
        "--strategy": str,
        "--n-paths": doceasy.Use(int)
    }))

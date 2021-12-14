#!/usr/bin/env python3
"""Usage: split_traffic [options] INFILE HDF_FILE

Options:
    --standardize-sizes
        Standardizes size information.

    --use-ms
        Use "ms" unit when interpreting freq, rather than "packet" unit.

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


# nchoicse: number of "decision points" needed to choose paths
# npaths: # of potential paths to choose
# strategy: how to choose paths at each decision point
# current_p: for dwr strategy only, is the probability distribution to sample from
def choose_paths(nchoices, npaths, strategy, current_p=[]):
    if strategy == "none":
        return [0] * nchoices
    if strategy == "rr":
        paths = list(range(npaths)) * int(nchoices/npaths)
        return paths + list(range(nchoices % npaths))
    if strategy == "wr": # uniform random
        return random.choices(range(npaths), k=nchoices)
    if strategy == "136r": # fixed random
        if npaths == 3:
            return random.choices(range(npaths), weights=[0.1, 0.3, 0.6], k=nchoices)
        if npaths == 2:
            return random.choices(range(npaths), weights=[0.3, 0.7], k=nchoices)
    if strategy == "dwr":
        return random.choices(range(npaths), weights=current_p, k=nchoices)

def split_timestamp(timestamps, freq, use_ms):
    if not use_ms:
        return range(0, len(timestamps), freq)
    last_split = 0
    freq_s = freq / 1000.0
    split_indices = [0]
    for i, time in enumerate(timestamps):
        if time - last_split > freq_s:
            split_indices.append(i)
            while time - last_split > freq_s:
                last_split += freq_s
    return split_indices

def split(label, sizes, timestamps, strategy, freq, npaths, standard_size=False, use_ms=False):
    if len(sizes) <= npaths * freq:
        return [label], [sizes], [timestamps]
    new_labels = []
    new_sizes = []
    new_timestamps = []
    for n in range(npaths):
        new_sizes.append([])
        new_timestamps.append([])
    new_p = []
    if strategy == "dwr":
        new_p = np.random.dirichlet(np.ones(npaths), size=1)[0]

    split_indices = split_timestamp(timestamps, freq, use_ms)
    paths = choose_paths(len(split_indices), npaths, strategy, new_p)
    for i, split_index in enumerate(split_indices):
        if i == len(split_indices) - 1:
            next_split = len(sizes)
        else:
            next_split = split_indices[i+1]
        path = paths[i]
        new_sizes[path] += list(sizes[split_index:next_split])
        new_timestamps[path] += list(timestamps[split_index:next_split])

    # Normalize all new timestamp paths to 0, remove empty or paths that are too small
    new_timestamps = [
            [time - timestamp[0] for time in timestamp] for timestamp in new_timestamps
            if len(timestamp) > 10
        ]
    if standard_size:
        new_sizes = [np.sign(sizes) for sizes in new_sizes if len(sizes) > 10]
    else:
        new_sizes = [sizes for sizes in new_sizes if len(sizes) > 10]

    for n in range(len(new_sizes)):
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
    if strategy == "none":
        freq = 1
        npaths = 1
    else:
        freq = splitter_kw.pop("freq")
        npaths = splitter_kw.pop("n_paths")
    standard_sizes = splitter_kw.pop("standardize_sizes")
    use_ms = splitter_kw.pop("use_ms")


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
            l, s, t = split(labels[i], h5in["sizes"][i], h5in["timestamps"][i], strategy, freq, npaths, standard_sizes, use_ms)
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
        "--standardize-sizes": bool,
        "--use-ms": bool,
        "--freq": doceasy.Use(int),
        "--strategy": str,
        "--n-paths": doceasy.Use(int)
    }))

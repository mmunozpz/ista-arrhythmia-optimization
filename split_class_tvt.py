# codigo del splitting para classification, train/val + test

import numpy as np
import argparse
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder


def split_trainval_test(input_file, trainval_output, test_output, test_ratio=0.15, seed=42):
    print(f"Loading dataset from: {input_file}")
    dataset = np.load(input_file, allow_pickle=True).item()
    ecgs, labels, ids = np.array(dataset["ecgs"]), np.array(
        dataset["labels"]), np.array(dataset["ids"])

    print(
        f"Total samples: {len(ecgs)} | Unique ECG IDs: {len(np.unique(ids))}")

    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)

    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(gss.split(ecgs, labels_enc, groups=ids))

    trainval_data = {"ecgs": ecgs[trainval_idx],
                     "labels": labels_enc[trainval_idx], "ids": ids[trainval_idx]}
    test_data = {"ecgs": ecgs[test_idx],
                 "labels": labels_enc[test_idx],     "ids": ids[test_idx]}

    np.save(trainval_output, trainval_data, allow_pickle=True)
    np.save(test_output, test_data, allow_pickle=True)

    label_map_path = test_output.replace(".npy", "_label_encoder.npy")
    np.save(label_map_path, le.classes_, allow_pickle=True)

    print(
        f" Saved train/val set: {trainval_output} ({len(trainval_idx)} samples, {len(np.unique(ids[trainval_idx]))} IDs)")
    print(
        f" Saved test set:     {test_output} ({len(test_idx)} samples, {len(np.unique(ids[test_idx]))} IDs)")
    print(f" Label encoder saved: {label_map_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split ECG dataset into trainval (85%) and test (15%) by groups (ECG IDs)")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--trainval-output", type=str, required=True)
    parser.add_argument("--test-output", type=str, required=True)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_trainval_test(args.input_file, args.trainval_output,
                        args.test_output, args.test_ratio, args.seed)

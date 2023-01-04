import os
import sys
import pandas as pd
import time
import json

from cl_benchmarks_fewshot import get_dataframe as get_dataframe_fewshot

from autogluon.multimodal import MultiModalPredictor
from sklearn.model_selection import StratifiedKFold

import argparse

import logging
# logging.basicConfig(level=logging.NOTSET)
logging.getLogger("pytorch_lighting").setLevel("WARNING")
logging.disable(logging.INFO)
logger = logging.getLogger("automm")
# logger.propagate = False
logger.setLevel("WARNING")
logger.disabled = True

def get_bayer():
    class_csv = "/home/ubuntu/data/cl_datasets/annotations/bayer_classes.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/annotations/bayer_train_annotations.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/annotations/bayer_test_annotations.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/bayer/train/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/bayer/test/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_belgalogos(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/belgalogos_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/belgalogos_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/belgalogos_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/belgalogos/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/belgalogos/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/belgalogos/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/belgalogos/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/belgalogos/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_bijou_dogs():
    class_csv = "/home/ubuntu/data/cl_datasets/annotations/bijou_dogs_classes.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/annotations/bijou_dogs_train_annotations.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/annotations/bijou_dogs_test_annotations.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/bijou_dogs/train/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/bijou_dogs/test/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_cub200(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/cub200_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/cub200_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/cub200_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/cub200/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/cub200/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/cub200/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/cub200/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/cub200/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_dtd47(fewshot: bool = False):
    if fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/describabletextures_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/describabletextures_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/describabletextures_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/describabletextures/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/describabletextures/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/describabletextures/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/describabletextures/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/describabletextures/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_eflooddepth(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/europeanflooddepth_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/europeanflooddepth_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/europeanflooddepth_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/europeanflooddepth/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/europeanflooddepth/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/europeanflooddepth/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/europeanflooddepth/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/europeanflooddepth/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_fgvc_aircrafts(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/fgvc_aircrafts_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/fgvc_aircrafts_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/fgvc_aircrafts_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/fgvc_aircrafts/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/fgvc_aircrafts/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/fgvc_aircrafts/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/fgvc_aircrafts/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/fgvc_aircrafts/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_kindle():
    class_csv = "/home/ubuntu/data/cl_datasets/annotations/kindle_classes.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/annotations/kindle_train_annotations.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/annotations/kindle_test_annotations.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/kindle/train/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/kindle/test/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_magnetictiledefects(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/magnetictiledefects_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/magnetictiledefects_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/magnetictiledefects_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/magnetictiledefects/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/magnetictiledefects/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/magnetictiledefects/test.csv"
    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/magnetictiledefects/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/magnetictiledefects/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_nike(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/nike_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/nike_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/nike_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/nike/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/nike/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/nike/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/nike/train/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/nike/test/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_food101(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/food101_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/food101_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/food101_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/food101/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/food101/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/food101/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/food101/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/food101/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_ifood(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/ifood_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/ifood_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/ifood_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/ifood/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/ifood/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/ifood/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/ifood/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/ifood/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_minc2500(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/opensurfacesminc2500_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/opensurfacesminc2500_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/opensurfacesminc2500_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/opensurfacesminc2500/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/opensurfacesminc2500/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/opensurfacesminc2500/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/opensurfacesminc2500/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/opensurfacesminc2500/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_nwpu_resisc45(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/nwpu_resisc45_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/nwpu_resisc45_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/nwpu_resisc45_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/nwpu_resisc45/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/nwpu_resisc45/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/nwpu_resisc45/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/nwpu_resisc45/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/nwpu_resisc45/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_stanforddogs(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/stanforddogs_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/stanforddogs_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/stanforddogs_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/stanforddogs/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/stanforddogs/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/stanforddogs/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/stanforddogs/train/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/stanforddogs/test/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_stanfordcars(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/stanfordcars_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/stanfordcars_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/stanfordcars_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/stanfordcars/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/stanfordcars/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/stanfordcars/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/stanfordcars/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/stanfordcars/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_semartschool(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/semartschool_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/semartschool_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/semartschool_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/semartschool/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/semartschool/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/semartschool/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/semartschool/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/semartschool/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_malariacellimages(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/malariacellimages_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/malariacellimages_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/malariacellimages_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/malariacellimages/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/malariacellimages/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/malariacellimages/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/malariacellimages/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/malariacellimages/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_mit67(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/mit67_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/mit67_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/mit67_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/mit67/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/mit67/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/mit67/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/mit67/train/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/mit67/test/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_realogy():
    class_csv = "/home/ubuntu/data/cl_datasets/annotations/realogy_classification_prod_finetune_classes.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/annotations/realogy_classification_prod_finetune_train_annotations.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/annotations/realogy_classification_prod_finetune_test_annotations.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/realogy_classification_prod_finetune/train/" + train_df[
        "ImageID"
    ].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/realogy_classification_prod_finetune/test/" + test_df[
        "ImageID"
    ].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_oxfordflowers(fewshot: bool = False):
    if not fewshot:
        class_csv = "/home/ubuntu/data/cl_datasets/annotations/oxfordflowers_classes.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/annotations/oxfordflowers_train_annotations.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/annotations/oxfordflowers_test_annotations.csv"
    else:
        class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/oxfordflowers/class_map.csv"
        train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/oxfordflowers/train.csv"
        test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/oxfordflowers/test.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/oxfordflowers/train/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/oxfordflowers/test/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_redfin():
    class_csv = "/home/ubuntu/data/cl_datasets/annotations/redfin_classes.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/annotations/redfin_train_annotations.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/annotations/redfin_test_annotations.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/redfin/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/redfin/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_herbariumsmallv2():
    class_csv = "/home/ubuntu/data/cl_datasets/annotations/herbariumsmall_classes.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/annotations/herbariumsmallv2_train_annotations.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/annotations/herbariumsmallv2_test_annotations.csv"

    classes_df_raw = pd.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/herbariumsmallv2/train/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/herbariumsmallv2/test/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_mnist():
    train_csv = "/home/ubuntu/data/ap_datasets/mnist/mnist_train_anno.csv"
    test_csv = "/home/ubuntu/data/ap_datasets/mnist/mnist_val_anno.csv"
    number_classes = 10

    train_df_raw = pd.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    train_df["ImageID"] = "/home/ubuntu/data/ap_datasets/mnist/train/" + train_df["ImageID"].astype(str)

    test_df_raw = pd.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
    test_df["ImageID"] = "/home/ubuntu/data/ap_datasets/mnist/test/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_dataframe(dataset_name, fewshot: bool = False):
    if dataset_name == "bayer":
        train_df, test_df = get_bayer()
    elif dataset_name == "belgalogos":
        train_df, test_df = get_belgalogos(fewshot=fewshot)
    elif dataset_name == "bijou_dogs":
        train_df, test_df = get_bijou_dogs()
    elif dataset_name == "cub200":
        train_df, test_df = get_cub200(fewshot=fewshot)
    elif dataset_name == "dtd47":
        train_df, test_df = get_dtd47(fewshot=fewshot)
    elif dataset_name == "eflooddepth":
        train_df, test_df = get_eflooddepth(fewshot=fewshot)
    elif dataset_name == "fgvc_aircrafts":
        train_df, test_df = get_fgvc_aircrafts(fewshot=fewshot)
    elif dataset_name == "kindle":
        train_df, test_df = get_kindle()
    elif dataset_name == "magnetictiledefects":
        train_df, test_df = get_magnetictiledefects(fewshot=fewshot)
    elif dataset_name == "nike":
        train_df, test_df = get_nike(fewshot=fewshot)
    elif dataset_name == "food101":
        train_df, test_df = get_food101(fewshot=fewshot)
    elif dataset_name == "ifood":
        train_df, test_df = get_ifood(fewshot=fewshot)
    elif dataset_name == "minc2500":
        train_df, test_df = get_minc2500(fewshot=fewshot)
    elif dataset_name == "nwpu_resisc45":
        train_df, test_df = get_nwpu_resisc45(fewshot=fewshot)
    elif dataset_name == "stanforddogs":
        train_df, test_df = get_stanforddogs(fewshot=fewshot)
    elif dataset_name == "stanfordcars":
        train_df, test_df = get_stanfordcars(fewshot=fewshot)
    elif dataset_name == "semartschool":
        train_df, test_df = get_semartschool(fewshot=fewshot)
    elif dataset_name == "malariacellimages":
        train_df, test_df = get_malariacellimages(fewshot=fewshot)
    elif dataset_name == "mit67":
        train_df, test_df = get_mit67(fewshot=fewshot)
    elif dataset_name == "realogy":
        train_df, test_df = get_realogy()
    elif dataset_name == "oxfordflowers":
        train_df, test_df = get_oxfordflowers(fewshot=fewshot)
    elif dataset_name == "redfin":
        train_df, test_df = get_redfin()
    elif dataset_name == "herbariumsmall":
        train_df, test_df = get_herbariumsmallv2()
    elif dataset_name == "mnist":
        train_df, test_df = get_mnist()
    else:
        raise Exception("We don't support dataset %s at this moment. " % (dataset_name))
    return train_df, test_df


def automm_cl(
    dataset: str = "mnist",
    fewshot: bool = False,
    skf: StratifiedKFold = None,
    folds: int = None,
    mode: str = "standard",
):
    dataset_name = dataset
    if fewshot:
        train_df, test_df = get_dataframe_fewshot(dataset_name)
    else:
        train_df, test_df = get_dataframe(dataset_name, fewshot=False)
    print(train_df)
    print(train_df["ImageID"].iloc[0], f"exist: {os.path.exists(train_df['ImageID'].iloc[0])}")
    print(test_df)
    print(test_df["ImageID"].iloc[0], f"exist: {os.path.exists(test_df['ImageID'].iloc[0])}")
    # print(test_df["ImageID"].iloc[0])
    # exit()

    start_time = time.time()
    # Default AutoMM quick mode, mobilenetv3_large_100 model for 10 minutes
    if mode == "quick":
        print(f"using quick build")
        hyperparameters = {
            "env.per_gpu_batch_size": 32,
            "optimization.learning_rate": 1.0e-3,
            "optimization.optim_type": "adamw",
            "optimization.weight_decay": 1.0e-3,
            "model.timm_image.checkpoint_name": "mobilenetv3_large_100",
        }
    elif mode == "standard":
        print(f"using standard build")
        hyperparameters = {
            "env.per_gpu_batch_size": 32,
            "optimization.max_epochs": 10,
            "optimization.learning_rate": 1.0e-4,
            "optimization.optim_type": "adamw",
            "optimization.weight_decay": 1.0e-3,
            "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224",
        }
    else:
        raise Exception(f"mode {mode} is not currently supported")

    if skf is not None:
        assert folds is not None, f"Expected folds to be not None"
        assert folds > 1, f"Expected folds to be larger than 1, but got folds={folds}"
        # Do cross validation
        y_train = train_df[args.label_name]
        X_train = train_df.drop(args.label_name, axis=1)
        highest_score = 0
        best_predictor = None
        for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

            train_df = pd.concat([X_t, y_t], axis=1)
            val_df = pd.concat([X_v, y_v], axis=1)

            predictor = MultiModalPredictor(label="LabelName", problem_type="classification", eval_metric="acc")

            predictor.fit(
                train_data=train_df,
                #   tuning_data=train_df,  # hacky solution for fewshot datasets: bijou_dogs, redfin
                hyperparameters=hyperparameters,
                time_limit=600 if mode == "quick" else None,
            )

            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"Elapsed time for fold {i} is {elapsed_time:d} seconds")

            scores = predictor.evaluate(val_df, metrics=["accuracy"])
            # print('Top-1 val acc: %.3f' % scores["accuracy"])
            print(f"Top-1 val acc for fold {i}: {scores['accuracy']:.3f}")

            start_time = time.time()

            # get best predictor
            if scores["accuracy"] > highest_score:
                highest_score = scores["accuracy"]
                best_predictor = predictor

        assert best_predictor is not None, "Expected best_predictor to be not None!"
        scores = best_predictor.evaluate(test_df, metrics=["accuracy"])
        print("Top-1 test acc: %.3f" % scores["accuracy"])

    else:
        predictor = MultiModalPredictor(label="LabelName", problem_type="classification", eval_metric="acc")
        predictor.fit(
            train_data=train_df,
            # presets="clip_swin_large_fusion",
            tuning_data=test_df,  # hacky solution for fewshot datasets: bijou_dogs, redfin
            # config={
            #     "model.names": [
            #         "clip",
            #         "timm_image",
            #         "fusion_mlp"
            #     ],
            #     "model.clip.checkpoint_name": "openai/clip-vit-large-patch14-336",
            #     "model.timm_image.checkpoint_name": "swin_large_patch4_window7_224",
            #     "model.clip.max_text_len": 0,
            #     "env.num_workers": 2,
            # },
            hyperparameters=hyperparameters,
            time_limit=600 if mode == "quick" else None,
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time is %d seconds" % elapsed_time)

        scores = predictor.evaluate(test_df, metrics=["accuracy"])
        print("Top-1 test acc: %.3f" % scores["accuracy"])


if __name__ == "__main__":
    # import sys
    # dataset = sys.argv[1]

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--cross-validate", action="store_true", default=False)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--label-name", type=str, default="LabelName", help="The column name of predictive label.")
    parser.add_argument("--logdir", type=str, help="Log dir")
    parser.add_argument("--fewshot", action="store_true", default=False)


    args = parser.parse_args()
    print(args.__dict__)
    args_save_path = os.path.join(args.logdir, f"{args.dataset}_config.json")
    print(f"saving args to {args_save_path}")
    json.dump(args.__dict__, open(args_save_path, "w"))

    mode = "fewshot" if args.fewshot else "fullshot"
    if args.cross_validate:
        mode += f"{args.folds}+cross_validation"
    else:
        mode += "+standard"
    print(f"Running benchmark {args.dataset} in {mode} mode")

    if args.cross_validate:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True)
        automm_cl(dataset=args.dataset, fewshot=args.fewshot, skf=skf, folds=args.folds)
    else:
        automm_cl(dataset=args.dataset, fewshot=args.fewshot, skf=None, folds=None)

import os
import sys
import pandas
import time

from autogluon.vision import ImageDataset
from autogluon.multimodal import MultiModalPredictor


def get_bayer():
    class_csv = "/home/ubuntu/data/cl_datasets/annotations/bayer_classes.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/annotations/bayer_train_annotations.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/annotations/bayer_test_annotations.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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

    test_df_raw = pandas.read_csv(test_csv)
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


def get_belgalogos():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/belgalogos/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/belgalogos/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/belgalogos/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_bijou_dogs():
    class_csv = "/home/ubuntu/data/cl_datasets/annotations/bijou_dogs_classes.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/annotations/bijou_dogs_train_annotations.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/annotations/bijou_dogs_test_annotations.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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

    test_df_raw = pandas.read_csv(test_csv)
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


def get_cub200():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/cub200/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/cub200/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/cub200/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_dtd47():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/describabletextures/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/describabletextures/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/describabletextures/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_eflooddepth():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/europeanflooddepth/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/europeanflooddepth/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/europeanflooddepth/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_fgvc_aircrafts():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/fgvc_aircrafts/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/fgvc_aircrafts/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/fgvc_aircrafts/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_kindle():
    class_csv = "/home/ubuntu/data/cl_datasets/annotations/kindle_classes.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/annotations/kindle_train_annotations.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/annotations/kindle_test_annotations.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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

    test_df_raw = pandas.read_csv(test_csv)
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


def get_magnetictiledefects():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/magnetictiledefects/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/magnetictiledefects/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/magnetictiledefects/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_nike():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/nike/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/nike/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/nike/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_food101():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/food101/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/food101/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/food101/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_ifood():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/ifood/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/ifood/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/ifood/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_minc2500():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/opensurfacesminc2500/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/opensurfacesminc2500/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/opensurfacesminc2500/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_nwpu_resisc45():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/nwpu_resisc45/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/nwpu_resisc45/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/nwpu_resisc45/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_stanforddogs():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/stanforddogs/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/stanforddogs/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/stanforddogs/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % number_classes)
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_stanfordcars():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/stanfordcars/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/stanfordcars/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/stanfordcars/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_semartschool():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/semartschool/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/semartschool/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/semartschool/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_malariacellimages():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/malariacellimages/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/malariacellimages/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/malariacellimages/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_mit67():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/mit67/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/mit67/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/mit67/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_realogy():
    class_csv = "/home/ubuntu/data/cl_datasets/annotations/realogy_classification_prod_finetune_classes.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/annotations/realogy_classification_prod_finetune_train_annotations.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/annotations/realogy_classification_prod_finetune_test_annotations.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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

    test_df_raw = pandas.read_csv(test_csv)
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


def get_oxfordflowers():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/oxfordflowers/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/oxfordflowers/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/oxfordflowers/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_redfin():
    class_csv = "/home/ubuntu/data/cl_datasets/annotations/redfin_classes.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/annotations/redfin_train_annotations.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/annotations/redfin_test_annotations.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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

    test_df_raw = pandas.read_csv(test_csv)
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

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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

    test_df_raw = pandas.read_csv(test_csv)
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


def get_cucumber():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/cucumber/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/cucumber/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/cucumber/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_icassava():
    class_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/icassava/class_map.csv"
    train_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/icassava/train.csv"
    test_csv = "/home/ubuntu/data/cl_datasets/fewshot_annotations/icassava/test.csv"

    classes_df_raw = pandas.read_csv(class_csv, header=None, sep="\t")
    number_classes = len(classes_df_raw)

    train_df_raw = pandas.read_csv(train_csv)
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
    train_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + train_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
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
    test_df["ImageID"] = "/home/ubuntu/data/cl_datasets/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, test_df


def get_dataframe(dataset_name):
    if dataset_name == "bayer":
        train_df, test_df = get_bayer()
    elif dataset_name == "belgalogos":
        train_df, test_df = get_belgalogos()
    elif dataset_name == "bijou_dogs":
        train_df, test_df = get_bijou_dogs()
    elif dataset_name == "cub200":
        train_df, test_df = get_cub200()
    elif dataset_name == "dtd47":
        train_df, test_df = get_dtd47()
    elif dataset_name == "eflooddepth":
        train_df, test_df = get_eflooddepth()
    elif dataset_name == "fgvc_aircrafts":
        train_df, test_df = get_fgvc_aircrafts()
    elif dataset_name == "kindle":
        train_df, test_df = get_kindle()
    elif dataset_name == "magnetictiledefects":
        train_df, test_df = get_magnetictiledefects()
    elif dataset_name == "nike":
        train_df, test_df = get_nike()
    elif dataset_name == "food101":
        train_df, test_df = get_food101()
    elif dataset_name == "ifood":
        train_df, test_df = get_ifood()
    elif dataset_name == "minc2500":
        train_df, test_df = get_minc2500()
    elif dataset_name == "nwpu_resisc45":
        train_df, test_df = get_nwpu_resisc45()
    elif dataset_name == "stanforddogs":
        train_df, test_df = get_stanforddogs()
    elif dataset_name == "stanfordcars":
        train_df, test_df = get_stanfordcars()
    elif dataset_name == "semartschool":
        train_df, test_df = get_semartschool()
    elif dataset_name == "malariacellimages":
        train_df, test_df = get_malariacellimages()
    elif dataset_name == "mit67":
        train_df, test_df = get_mit67()
    elif dataset_name == "realogy":
        train_df, test_df = get_realogy()
    elif dataset_name == "oxfordflowers":
        train_df, test_df = get_oxfordflowers()
    elif dataset_name == "redfin":
        train_df, test_df = get_redfin()
    elif dataset_name == "herbariumsmall":
        train_df, test_df = get_herbariumsmallv2()
    elif dataset_name == "cucumber":
        train_df, test_df = get_cucumber()
    elif dataset_name == "icassava":
        train_df, test_df = get_icassava()
    else:
        raise Exception("We don't support dataset %s at this moment. " % (dataset_name))
    return train_df, test_df


def automm_cl():
    dataset_name = "icassava"
    train_df, test_df = get_dataframe(dataset_name)
    print(train_df, test_df)

    start_time = time.time()
    # Default AutoMM, Swin-base model train 10 epochs
    hyperparameters = {
        "env.per_gpu_batch_size": 32,
        "optimization.max_epochs": 10,
    }

    predictor = MultiModalPredictor(label="LabelName", problem_type="classification", eval_metric="acc")
    predictor.fit(
        train_data=train_df,
        tuning_data=test_df,  # hacky solution for fewshot datasets: bijou_dogs, redfin
        hyperparameters=hyperparameters,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    scores = predictor.evaluate(test_df, metrics=["accuracy"])
    print("Top-1 test acc: %.3f" % scores["accuracy"])
    print("Elapsed time is %d seconds" % elapsed_time)


if __name__ == "__main__":
    automm_cl()

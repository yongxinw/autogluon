import os
import sys
import pandas
import time

from autogluon.multimodal import MultiModalPredictor


def get_caltech256():
    train_csv = "/home/ubuntu/data/ap_datasets/Caltech256/caltech256_train_anno.csv"
    val_csv = "/home/ubuntu/data/ap_datasets/Caltech256/caltech256_val_anno.csv"
    test_csv = "/home/ubuntu/data/ap_datasets/Caltech256/caltech256_test_anno.csv"
    number_classes = 257

    train_df_raw = pandas.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
                 "IsDepiction", "IsInside"])
    train_df['ImageID'] = "/home/ubuntu/data/ap_datasets/Caltech256/train/" + train_df["ImageID"].astype(str)

    val_df_raw = pandas.read_csv(val_csv)
    val_df = val_df_raw.drop(
        columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
                 "IsDepiction", "IsInside"])
    val_df['ImageID'] = "/home/ubuntu/data/ap_datasets/Caltech256/validation/" + val_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
                 "IsDepiction", "IsInside"])
    test_df['ImageID'] = "/home/ubuntu/data/ap_datasets/Caltech256/test/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The validation set has %d samples" % (len(val_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, val_df, test_df


def get_oxfordiiipet():
    train_csv = "/home/ubuntu/data/ap_datasets/OxfordIIITPet/OxfordIIITPet_train_anno.csv"
    val_csv = "/home/ubuntu/data/ap_datasets/OxfordIIITPet/OxfordIIITPet_val_anno.csv"
    test_csv = "/home/ubuntu/data/ap_datasets/OxfordIIITPet/OxfordIIITPet_test_anno.csv"
    number_classes = 37

    train_df_raw = pandas.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
                 "IsDepiction", "IsInside"])
    train_df['ImageID'] = "/home/ubuntu/data/ap_datasets/OxfordIIITPet/train/" + train_df["ImageID"].astype(str)

    val_df_raw = pandas.read_csv(val_csv)
    val_df = val_df_raw.drop(
        columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
                 "IsDepiction", "IsInside"])
    val_df['ImageID'] = "/home/ubuntu/data/ap_datasets/OxfordIIITPet/validation/" + val_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
                 "IsDepiction", "IsInside"])
    test_df['ImageID'] = "/home/ubuntu/data/ap_datasets/OxfordIIITPet/test/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The validation set has %d samples" % (len(val_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, val_df, test_df


def get_sun397():
    train_csv = "/home/ubuntu/data/ap_datasets/SUN397/sun397_train_anno.csv"
    val_csv = "/home/ubuntu/data/ap_datasets/SUN397/sun397_val_anno.csv"
    test_csv = "/home/ubuntu/data/ap_datasets/SUN397/sun397_test_anno.csv"
    number_classes = 397

    train_df_raw = pandas.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
                 "IsDepiction", "IsInside"])
    train_df['ImageID'] = "/home/ubuntu/data/ap_datasets/SUN397/SUN397/" + train_df["ImageID"].astype(str)

    val_df_raw = pandas.read_csv(val_csv)
    val_df = val_df_raw.drop(
        columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
                 "IsDepiction", "IsInside"])
    val_df['ImageID'] = "/home/ubuntu/data/ap_datasets/SUN397/SUN397/" + val_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
                 "IsDepiction", "IsInside"])
    test_df['ImageID'] = "/home/ubuntu/data/ap_datasets/SUN397/SUN397/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The validation set has %d samples" % (len(val_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, val_df, test_df


def get_cifar10():
    train_csv = "/home/ubuntu/data/ap_datasets/CIFAR10/cifar10_train_anno.csv"
    val_csv = "/home/ubuntu/data/ap_datasets/CIFAR10/cifar10_val_anno.csv"
    test_csv = "/home/ubuntu/data/ap_datasets/CIFAR10/cifar10_test_anno.csv"
    number_classes = 10

    train_df_raw = pandas.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
                 "IsDepiction", "IsInside"])
    train_df['ImageID'] = "/home/ubuntu/data/ap_datasets/CIFAR10/train/" + train_df["ImageID"].astype(str)

    val_df_raw = pandas.read_csv(val_csv)
    val_df = val_df_raw.drop(
        columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
                 "IsDepiction", "IsInside"])
    val_df['ImageID'] = "/home/ubuntu/data/ap_datasets/CIFAR10/validation/" + val_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
                 "IsDepiction", "IsInside"])
    test_df['ImageID'] = "/home/ubuntu/data/ap_datasets/CIFAR10/test/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The validation set has %d samples" % (len(val_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, val_df, test_df


def get_cifar100():
    train_csv = "/home/ubuntu/data/ap_datasets/CIFAR100/cifar100_train_anno.csv"
    val_csv = "/home/ubuntu/data/ap_datasets/CIFAR100/cifar100_val_anno.csv"
    test_csv = "/home/ubuntu/data/ap_datasets/CIFAR100/cifar100_test_anno.csv"
    number_classes = 257

    train_df_raw = pandas.read_csv(train_csv)
    train_df = train_df_raw.drop(
        columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
                 "IsDepiction", "IsInside"])
    train_df['ImageID'] = "/home/ubuntu/data/ap_datasets/CIFAR100/train/" + train_df["ImageID"].astype(str)

    val_df_raw = pandas.read_csv(val_csv)
    val_df = val_df_raw.drop(
        columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
                 "IsDepiction", "IsInside"])
    val_df['ImageID'] = "/home/ubuntu/data/ap_datasets/CIFAR100/validation/" + val_df["ImageID"].astype(str)

    test_df_raw = pandas.read_csv(test_csv)
    test_df = test_df_raw.drop(
        columns=["Source", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
                 "IsDepiction", "IsInside"])
    test_df['ImageID'] = "/home/ubuntu/data/ap_datasets/CIFAR100/test/" + test_df["ImageID"].astype(str)

    print("This dataset has %d classes" % (number_classes))
    print("The training set has %d samples" % (len(train_df)))
    print("The validation set has %d samples" % (len(val_df)))
    print("The test set has %d samples" % (len(test_df)))
    return train_df, val_df, test_df


def get_dataframe(dataset_name):
    if dataset_name == "caltech256":
        train_df, val_df, test_df = get_caltech256()
    elif dataset_name == "cifar10":
        train_df, val_df, test_df = get_cifar10()
    elif dataset_name == "cifar100":
        train_df, val_df, test_df = get_cifar100()
    elif dataset_name == "oxfordiiipet":
        train_df, val_df, test_df = get_oxfordiiipet()
    elif dataset_name == "sun397":
        train_df, val_df, test_df = get_sun397()
    else:
        print("We don't support dataset %s at this moment. " % (dataset_name))
    return train_df, val_df, test_df


def automm_cl(dataset: str = "oxfordiiipet"):
    dataset_name = dataset
    train_df, val_df, test_df = get_dataframe(dataset_name)
    print(train_df, val_df, test_df)

    start_time = time.time()
    # Default AutoMM, Swin-base model train 10 epochs
    hyperparameters = {
        "env.per_gpu_batch_size": 32,
        "optimization.max_epochs": 10,
    }

    predictor = MultiModalPredictor(label="LabelName",
                                    problem_type="classification",
                                    eval_metric="acc")

    predictor.fit(train_data=train_df,
                  tuning_data=val_df,  # hacky solution for fewshot datasets: bijou_dogs, redfin
                  hyperparameters=hyperparameters)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Elapsed time is %d seconds' % elapsed_time)

    scores = predictor.evaluate(test_df, metrics=["accuracy"])
    print('Top-1 test acc: %.3f' % scores["accuracy"])


if __name__ == "__main__":
    # caltech256
    # cifar10
    # cifar100
    # oxfordiiipet
    # sun397
    import sys
    dataset = sys.argv[1]
    automm_cl(dataset=dataset)

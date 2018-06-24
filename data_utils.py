import cv2
import os
import argparse


def labelDicFromFile(name):
    label_dic = {}
    with open(name) as f:
        for line in f:
            (val, key) = line.split()
            label_dic[key] = int(val)
    return label_dic

def dirToVideoLabel(data_dir, label_dic):
    labels = []
    filenames = []
    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)
        for video in os.listdir(label_dir):
            videoFile = os.path.join(label_dir, video)
            filenames.append(videoFile)
            labels.append(int(label_dic[label_name]) - 1)
    return filenames, labels



def loadPaths(data_dir="/data/nvidia-docker/data", dataset='UCF-101', train_set_number='all', test_set_number='all'):
    """
    Load the data

    Arguments:
    dataset -- the name of the dataset
    train_set_number -- '1', '2', '3' or 'all'
    test_set_number -- '1', '2', '3' or 'all'
    """

    # TODO: Add a check with some warning message if the dataset is not found
    res = {
        "train": {
            "paths": [],
            "labelnames": [],
            "labels": []
        },
        "test": {
            "paths": [],
            "labelnames": [],
            "labels": []
        }
    }
    ucf_lists = os.path.join(data_dir,'ucfTrainTestlist')
    ucf101 = os.path.join(data_dir,'UCF-101')
    classMapFile = os.path.join(data_dir,"ucfTrainTestlist/classInd.txt")
    if (dataset == 'UCF-101'):
        label_dic = labelDicFromFile(classMapFile)
        train_set_names = []
        test_set_names = []
        if (train_set_number != 'all'):
            train_set_names.append('trainlist0{}.txt'.format(train_set_number))
        else:
            [train_set_names.append('trainlist0{}.txt'.format(i)) for i in range(1,4)]
        if (test_set_number != 'all'):
            test_set_names.append('testlist0{}.txt'.format(test_set_number))
        else:
            [test_set_names.append('testlist0{}.txt'.format(i)) for i in range(1,4)]

        for train_set_name in train_set_names:
            train_list = open('{}/{}'.format(ucf_lists,train_set_name), 'r')
            num = os.path.splitext(train_set_name)[0].split('list')[1]
            for line in train_list:
                filepath = line.split(' ')[0]
                label = filepath.split("/")[0]
                res["train"]["paths"].append(os.path.join(data_dir,'UCF-101',filepath))
                res["train"]["labelnames"].append(label)
                res["train"]["labels"].append(label_dic[label]-1)



        for test_set_name in test_set_names:
            test_list = open('{}/{}'.format(ucf_lists,test_set_name), 'r')
            num = os.path.splitext(test_set_name)[0].split('list')[1]
            # test_dir = '{}_test{}'.format(ucf101, num)
            for line in test_list:
                filepath = line[:-1]
                label = filepath.split("/")[0]
                res["test"]["paths"].append(os.path.join(data_dir,'UCF-101',filepath))
                res["test"]["labelnames"].append(label)
                res["test"]["labels"].append(label_dic[label]-1)
        return res
# r  = loadPaths()
# for i in zip(r["train"]["labels"],r["train"]["paths"],r["train"]["labelnames"]):
#     print(i)
# trains
# tests
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='?', const='UCF-101',
                        default='UCF-101')
    parser.add_argument('--train_number', nargs = '?', const='all',
                        default = 'all')
    parser.add_argument('--test_number', nargs = '?', const='all',
                        default = 'all')
    args = parser.parse_args()

    dataset = args.dataset
    train_number = args.train_number
    test_number = args.test_number
    load_data(dataset, train_number, test_number)

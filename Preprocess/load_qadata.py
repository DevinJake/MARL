# -*- coding: utf-8 -*-
# @Time    : 2019/1/20 14:50

import re

import fnmatch
import os


def load_qadata(qa_dir):
    """

    :param qa_dir: 预处理后的qa数据的文件夹，eg：/home/zhangjingyao/preprocessed_data_10k/test
    :return: 这个文件夹下面，问答数据的字典。最外层是序号: QA_1_
    """
    print("begin_load_qadata")
    qa_set = {}
    for root, dirnames, filenames in os.walk(qa_dir):
        if(dirnames == []):
            qa_id = root[root.rfind("_")+1:]
            qa_dict ={}
            for filenames in fnmatch.filter(filenames, '*.txt'):
                pattern = re.compile('QA_\d+_')
                keystr = re.sub(pattern,"", filenames).replace(".txt","")
                qa_dict[keystr] = open(root+"/"+filenames).readlines()
            qa_set[qa_id] = qa_dict
    print("load_qadata_success")
    return qa_set

if __name__=="__main__":
    qa_set = load_qadata("/home/zhangjingyao/preprocessed_data_10k/test")

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

def getQA_by_state(qa_set):
    print(0)
    qmap = {}
    pset = set([])
    for qafile in qa_set.itervalues():
        for qid in range(len(qafile["context"])):
            # 得到一个qa数据
            q = {k: v[qid] for k, v in qafile.items()}

            # 解析问句
            qstring = q["context_utterance"]
            qstate = q["state"]
            qrelations = q["context_relations"].split('|')
            if(qmap.has_key(qstate)):
                qmap[qstate].append(q)
            else:
                qmap[qstate] = [q]


    print(pset)
    return qmap

if __name__=="__main__":
    qa_set = load_qadata("/home/zhangjingyao/preprocessed_data_10k/test")
    qa_map = getQA_by_state(qa_set)
    #print(qa_map)
    for k in qa_map.iterkeys():
        print "----------------------------"
        print "----------------------------"
        print "----------------------------"
        print k
        for i in range(0,20):
            print i
            # for item in qa_map[k][i].iteritems():
            #     print item
            print 'state:',qa_map[k][i]['state'],
            print "context_utterance:",qa_map[k][i]['context_utterance'],
            print 'context_relations:',qa_map[k][i]['context_relations'],
            print 'context_entities:',qa_map[k][i]['context_entities'],
            print 'context_types:',qa_map[k][i]['context_types'],
            print 'context:',qa_map[k][i]['context'],
            print 'orig_response:',qa_map[k][i]['orig_response'],
            print 'response_entities:',qa_map[k][i]['response_entities'],
            print "----------------------------"
            print "SYMBOLIC:\n"
            print "----------------------------"
            print "CODE:\n"
            print "----------------------------"

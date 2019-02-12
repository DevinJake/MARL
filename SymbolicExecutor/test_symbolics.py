# -*- coding: utf-8 -*-
# @Time    : 2019/1/19 17:58
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import json
import random
import time
from unittest import TestCase
from urllib import urlencode

import requests

from Model.seq2seq import Seq2Seq
from Preprocess.load_qadata import load_qadata
from Preprocess.load_wikidata import load_wikidata
from Preprocess.question_parser import QuestionParser
from SymbolicExecutor.symbolics import Symbolics

from params import get_params


class TestSymbolics():

    def test_sparql(self,e='Q148',r = 'P17',t = 'Q4022'):
        answer_dict = {}
        anser_values = []
        sparql = {"query": "SELECT ?river WHERE { \
                                    ?river wdt:" + r + " wd:"+ e +". \
                                    ?river wdt:P31  wd:"+ t + ". \
                               }",
                  "format" : "json",
                  }
        print sparql
        sparql =  urlencode(sparql)
        print sparql
        url = 'https://query.wikidata.org/sparql?'+sparql
        r = requests.get(url)
        #print r.json()["results"]
        for e in r.json()["results"]["bindings"]:
            entity =  e["river"]["value"].split("/")[-1]
            anser_values.append(entity)
        answer_dict[e] = anser_values

    def test_select(self):
        params = get_params("/data/zjy/csqa_data", "/home/zhangjingyao/preprocessed_data_10k")

        # ls = LuceneSearch(params["lucene_dir"])
        # 读取知识库
        wikidata, item_data, prop_data, child_par_dict = load_wikidata(params["wikidata_dir"])# data for entity ,property, type

        # 读取qa文件集
        qa_set = load_qadata("/home/zhangjingyao/preprocessed_data_10k/demo")
        question_parser = QuestionParser(params, True)

        f = open("log.txt", 'w+')
        for qafile in qa_set.itervalues():
            for qid in range(len(qafile["context"])):
                # 得到一个qa数据
                q = {k:v[qid] for k,v in qafile.items()}

                # 解析问句
                qstring = q["context_utterance"]
                entities = question_parser.getNER(q)
                relations = question_parser.getRelations(q)
                types = question_parser.getTypes(q)

                # 得到操作序列
                states = random.randint(13,13) # 随机生成操作序列
                seq2seq = Seq2Seq()
                symbolic_seq = seq2seq.simple(qstring,entities,relations,types, states)

                # 符号执行
                time_start = time.time()
                symbolic_exe = Symbolics(wikidata, item_data, prop_data, child_par_dict, symbolic_seq)
                answer = symbolic_exe.executor()

                print("answer is :", answer)

                time_end = time.time()
                print('time cost:', time_end - time_start)
                print("--------------------------------------------------------------------------------")


        print(0)
# #wikidata, reverse_dict, prop_data, child_par_dict, wikidata_fanout_dict = load_wikidata(param["wikidata_dir"])

if __name__ == "__main__":
    test = TestSymbolics().test_select()
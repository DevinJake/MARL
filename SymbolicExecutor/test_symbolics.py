# -*- coding: utf-8 -*-
# @Time    : 2019/1/19 17:58
# @Author  : Yaoleo
# @Blog    : yaoleo.github.io
import json
import random
from unittest import TestCase

from LuceneSearch import LuceneSearch
from Model.seq2seq import Seq2Seq
from Preprocess.load_qadata import load_qadata
from Preprocess.load_wikidata import load_wikidata
from Preprocess.question_parser import QuestionParser
from SymbolicExecutor.symbolics import Symbolics

from params import get_params


class TestSymbolics(TestCase):
    def test_select(self):
        params = get_params("/data/zjy/csqa_data", "/home/zhangjingyao/preprocessed_data_10k")

        # ls = LuceneSearch(params["lucene_dir"])
        # 读取知识库
        wikidata, item_data, prop_data, child_par_dict = load_wikidata(params["wikidata_dir"])# data for entity ,property, type

        # 读取qa文件集
        qa_set = load_qadata("/home/zhangjingyao/preprocessed_data_10k/demo")
        question_parser = QuestionParser(params, True)
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
                states = random.randint(1,8) # 随机生成操作序列
                seq2seq = Seq2Seq()
                symbolic_seq =seq2seq.simple(qstring,entities,relations,types, states)

                # 符号执行
                symbolic_exe = Symbolics(wikidata,item_data, prop_data, child_par_dict, symbolic_seq)
                result = symbolic_exe.executor()


        print(0)
# #wikidata, reverse_dict, prop_data, child_par_dict, wikidata_fanout_dict = load_wikidata(param["wikidata_dir"])
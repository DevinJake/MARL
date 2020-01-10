# -*- coding: utf-8 -*-
import json
import string
import os
from functools import cmp_to_key
import random

class Retriever_WebQSP():
    def __init__(self, dictwebqsp, dictwebqsp_weak):
        self.dictwebqsp = dictwebqsp
        self.dictwebqsp_weak = dictwebqsp_weak

        # The cache is used to store the retrieved samples for first time in memory.
        # Therefore in next iteration it is not needed to find support set in webqsp file.
        self.support_set_cache = {}

    def takequestion(self, dict_item, question):
        if len(dict_item) > 0:
            takequestionvalues = list(dict_item.values())[0]
            return self.CalculatesimilarityStr(takequestionvalues, question) * (-1)
        else:
            return 0.0

    def Retrieve(self, N, question):
        candidate_list = question
        sort_candidate = sorted(candidate_list, key=self.takequestion)

        # remove the quesiton itself
        for candidateItem in sort_candidate:
            if list(candidateItem.values())[0] == question:
                sort_candidate.remove(candidateItem)
                break

        topNList = sort_candidate if len(sort_candidate) <= N else sort_candidate[0:N]
        return topNList

    # The input of the model is constrained by the maximum number of tokens.
    # When finding top-N, it should considered whether the question in full training dateset
    # is removed from the model or not.
    def RetrieveWithMaxTokens(self, N, key_name, key_weak, question, train_data_webqsp, weak_flag, qid):
        if qid in self.support_set_cache:
            print('%s is in top-N cache!' %(str(qid)))
            return self.support_set_cache[qid]
        print('%s is not in top-N cache!' % (str(qid)))
        dict_candicate = self.dictwebqsp_weak
        topNList = list()
        if key_name in self.dictwebqsp:
            candidate_list = self.dictwebqsp[key_name]
            candidate_list_filtered = [x for x in candidate_list if len(x)>0 and list(x.keys())[0] in train_data_webqsp]
            sort_candidate = sorted(candidate_list_filtered, key=lambda x: self.takequestion(x, question))

            # remove the quesiton itself
            for candidateItem in sort_candidate:
                if list(candidateItem.values())[0] == question:
                    sort_candidate.remove(candidateItem)

            topNList = sort_candidate if len(sort_candidate) <= N else sort_candidate[0:N]

            # if don't have enough matches, search without relation match
            if len(topNList) < N and weak_flag:
                # print(len(topNList), " found of ", N)
                if key_weak in dict_candicate:
                    weak_list = dict_candicate[key_weak]
                    weak_list_filtered = [x for x in weak_list if len(x)>0 and list(x.keys())[0] in train_data_webqsp]
                    sort_candidate_weak = sorted(weak_list_filtered, key=lambda x: self.takequestion(x, question))
                    for c_weak in sort_candidate_weak:
                        if len(topNList) == N:
                            break
                        if c_weak not in topNList:
                            topNList.append(c_weak)
                            # print(len(topNList))
        self.support_set_cache[qid] = topNList
        return topNList

    def MoreSimilarity(self, sentence1, sentence2):
        return True
        # sim1 = self.Calculatesimilarity(sentence1, question)
        # sim2 = self.Calculatesimilarity(sentence2, question)
        # # print(question, sentence1, sim1)
        # # print(question, sentence2, sim2)
        # similarity_result = sim1 > sim2
        # return similarity_result

    def Calculatesimilarity(self, sentence1, sentence2):
        trantab = str.maketrans({key: None for key in string.punctuation})
        s1 = str(sentence1.values()).translate(trantab)
        s2 = sentence2.translate(trantab)
        list1 = s1.split(' ')
        list2 = s2.split(' ')
        intersec = set(list1).intersection(set(list2))
        union = set([])
        union.update(list1)
        union.update(list2)
        jaccard = float(len(intersec)) / float(len(union)) if len(union) != 0 else 0
        return jaccard

    def CalculatesimilarityStr(self, sentence1, sentence2):
        trantab = str.maketrans({key: None for key in string.punctuation})
        s1 = sentence1.translate(trantab)
        s2 = sentence2.translate(trantab)
        list1 = s1.split(' ')
        list2 = s2.split(' ')
        intersec = set(list1).intersection(set(list2))
        union = set([])
        union.update(list1)
        union.update(list2)
        jaccard = float(len(intersec)) / float(len(union)) if len(union) != 0 else 0
        return jaccard


if __name__ == "__main__":

    result_dict = {}
    with open("WebQSP.train.json", "r", encoding='UTF-8') as questions:
        load_dict = json.load(questions)
        questions = load_dict["Questions"]
        retriever = Retriever_WebQSP(questions, {})

        print(len(questions))
        for q in questions:
            question = q["ProcessedQuestion"]
            Answers = []
            id = q["QuestionId"]
            answerList = q["Parses"][0]["Answers"]
            for an in answerList:
                Answers.append(an['AnswerArgument'])
            sparql = q["Parses"][0]["Sparql"]

            retriever = Retriever_WebQSP(load_dict, {})
            topNlist = retriever.Retrieve(5, question)


        # with open('top5_4weak11.13.json', 'w', encoding='utf-8') as f:




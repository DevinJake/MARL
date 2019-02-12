# -*- coding: utf-8 -*-
# @Time    : 2019/1/18 14:52

from urllib import urlencode

import requests


class Symbolics():

    def __init__(self, wiki_data, item_data, property_data, child_par_dict, seq):
        self.wikidata = wiki_data
        #self.item_data = item_data
        #self.pid2prop = property_data
        self.type_data = child_par_dict
        self.seq = seq
        self.answer = {}

    def executor(self):
        for symbolic in self.seq:
            key = symbolic.keys()[0]
            e = symbolic[key][0]
            r = symbolic[key][1]
            t = symbolic[key][2]
            if ("A1" in symbolic):
                self.answer = self.select(e, r, t)
            elif ("A2" in symbolic):
                self.answer = self.select_all(e, r, t)
            elif ("A3" in symbolic):
                self.answer = self.is_bool(e)
            elif ("A9" in symbolic):
                self.answer = self.union(e, r, t)
            elif ("A10" in symbolic):
                self.answer = self.inter(e, r, t)
            elif ("A11" in symbolic):
                self.answer = self.diff(e, r, t)
            elif ("A12" in symbolic):
                self.answer = self.count()
            else:
                print("wrong symbolic")

        return self.answer

    def select(self, e, r, t):
        print("A1:", e, r, t)
        answer_dict = self.answer
        answer_values = []

        for key in self.wikidata.keys():
            if ("P31" in self.wikidata[key] and r in self.wikidata[key]):
                if (t in self.wikidata[key]["P31"]):
                    if (e in self.wikidata[key][r]):
                        # print "correct entity", self.item_data[key], self.item_data[e]
                        answer_values.append(key)
        answer_dict[e] = answer_values
        return answer_dict

    def select_all(self, et, r, t):
        print("A2:", et, r, t)
        answer_dict = self.answer
        answer_values = []

        for key in self.wikidata.keys():
            if ("P31" in self.wikidata[key] and r in self.wikidata[key]):
                if (t in self.wikidata[key]["P31"]):
                    for e in self.wikidata[key][r]:
                        if(self.type_data[e] == et): # todo 这里的类型是CSQA数据里的 并不标准 因为一个实体有多个类型 这里只有一个
                            # print e, self.item_data[e],self.type_data[e],self.item_data[self.type_data[e]]
                            if(e in answer_dict):
                                answer_dict[e].append(key)
                            else:
                                answer_dict[e] = [key]
        return answer_dict

    def is_bool(self, e):
        print("is_bool")
        for key in self.answer:
            if (e in self.answer[key]):
                return True
        return False

    def arg_min(self):
        pass

    def arg_max(self):
        pass

    def less_than(self, e):
        pass

    def greater_than(self, e):
        pass

    def union(self, e, r, t):
        print("A9:", e, r, t)
        answer_dict = self.answer
        answer_values = []

        for key in self.wikidata.keys():
            if ("P31" in self.wikidata[key] and r in self.wikidata[key]):
                if (t in self.wikidata[key]["P31"]):
                    if (e in self.wikidata[key][r]):
                        # print "correct entity", self.item_data[key], self.item_data[e]
                        answer_values.append(key)
        answer_dict[e] = answer_values
        # 进行 union 操作 todo 这里前面都和select部分一样 所以还是应该拆开？ union单独做 好处是union可以不止合并两个 字典里的都可以合并
        union_key = ""
        union_value = set([])
        for k, v in answer_dict.iteritems():
            union_key += k + "&"
            union_value = union_value | set(v)
        union_key = union_key[:-1]
        answer_dict.clear()
        answer_dict[union_key] = set(union_value)

        return answer_dict
        pass

    def inter(self, e, r, t):
        print("A10:", e, r, t)
        answer_dict = self.answer
        answer_values = []

        for key in self.wikidata.keys():
            if ("P31" in self.wikidata[key] and r in self.wikidata[key]):
                if (t in self.wikidata[key]["P31"]):
                    if (e in self.wikidata[key][r]):
                        # print "correct entity", self.item_data[key], self.item_data[e]
                        answer_values.append(key)
        answer_dict[e] = answer_values
        # 进行 inter 类似 union
        inter_key = ""
        inter_value = set([])
        for k, v in answer_dict.iteritems():
            inter_key += k + "&"
            if len(inter_value) > 0:
                inter_value = inter_value & set(v)
            else:
                inter_value = set(v)

        answer_dict.clear()
        inter_key = inter_key[:-1]
        answer_dict[inter_key] = set(inter_value)

        return answer_dict

    def diff(self, e, r, t):
        print("A11:", e, r, t)
        answer_dict = self.answer
        answer_values = []

        for key in self.wikidata.keys():
            if ("P31" in self.wikidata[key] and r in self.wikidata[key]):
                if (t in self.wikidata[key]["P31"]):
                    if (e in self.wikidata[key][r]):
                        # print "correct entity", self.item_data[key], self.item_data[e]
                        answer_values.append(key)
        answer_dict[e] = answer_values
        # 进行 diff 操作 类似 union
        diff_key = ""
        diff_value = set([])
        for k, v in answer_dict.iteritems():
            diff_key += k + "-"
            if len(diff_value) > 0:
                diff_value = diff_value - set(v)
            else:
                diff_value = set(v)

        answer_dict.clear()
        diff_key = diff_key[:-1]
        answer_dict[diff_key] = set(diff_value)

        return answer_dict

    def count(self):
        if (len(self.answer) == 1):
            return len(self.answer.values()[0])
        else:
            print("more than one items in answer")
            return len(self.answer.keys())
        pass

    def at_least(self, N):
        pass

    def at_most(self, N):
        pass

    def get_keys(self):
        pass

    def EOQ(self):
        pass


    ########################
    def select_sparql(self, e, r, t):  # use sparql
        answer_dict = {}
        anser_values = []
        sparql = {"query": "SELECT ?river WHERE { \
                                            ?river wdt:" + r + " wd:" + e + ". \
                                            ?river wdt:P31  wd:" + t + ". \
                                       }",
                  "format": "json",
                  }
        print sparql
        sparql = urlencode(sparql)
        print sparql
        url = 'https://query.wikidata.org/sparql?' + sparql
        r = requests.get(url)
        # print r.json()["results"]
        for e in r.json()["results"]["bindings"]:
            entity = e["river"]["value"].split("/")[-1]
            anser_values.append(entity)
        answer_dict[e] = anser_values

        return answer_dict

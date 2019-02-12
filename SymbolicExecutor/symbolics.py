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
                self.print_answer()
            elif ("A2" in symbolic):
                self.answer = self.select_all(e, r, t)
                self.print_answer()
            elif ("A3" in symbolic):
                self.answer = self.is_bool(e)
                self.print_answer()
            elif ("A4" in symbolic):
                self.answer = self.arg_min()
                self.print_answer()
            elif ("A5" in symbolic):
                self.answer = self.arg_max()
                self.print_answer()
            elif ("A6" in symbolic):
                self.answer = self.greater_than(e)
                self.print_answer()
            elif ("A7" in symbolic):
                self.answer = self.less_than(e)
                self.print_answer()
            elif ("A9" in symbolic):
                self.answer = self.union(e, r, t)
                self.print_answer()
            elif ("A8" in symbolic):
                self.answer = self.inter(e, r, t)
                self.print_answer()
            elif ("A10" in symbolic):
                self.answer = self.diff(e, r, t)
                self.print_answer()
            elif ("A11" in symbolic):
                self.answer = self.count()
                self.print_answer()
            elif ("A12" in symbolic):
                self.answer = self.at_least(e)
                self.print_answer()
            elif ("A13" in symbolic):
                self.answer = self.at_most(e)
                self.print_answer()
            elif ("A14" in symbolic):
                self.answer = self.equal(e)
                self.print_answer()
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
                if (et in self.wikidata[key]["P31"]):
                    for e in self.wikidata[key][r]:
                        if(e in self.wikidata and t in self.wikidata[e]["P31"]): # todo 这里的类型是CSQA数据里的 并不标准 因为一个实体有多个类型 这里只有一个
                            # print e, self.item_data[e],self.type_data[e],self.item_data[self.type_data[e]]
                            if(key in answer_dict):
                                answer_dict[key].append(e)
                            else:
                                answer_dict[key] = [e]
        return answer_dict

    def is_bool(self, e):
        print("A3: is_bool")
        for key in self.answer:
            if (e in self.answer[key]):
                return True
        return False

    def arg_min(self):
        print("A4: arg_min")
        if not self.answer:
            return None
        min = 99999
        min_k = self.answer.keys[0]
        for k, v in self.answer.iteritems():
            if len(v) < min:
                min = len(v)
                min_k = k
        return min_k

    def arg_max(self):
        print("A5: arg_max")
        if not self.answer:
            return None
        max = 0
        max_k, max_v = self.answer.popitem()
        for k,v in self.answer.iteritems():
            if len(v) > max:
                max = len(v)
                max_k = k
        return max_k

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
        print("A8:", e, r, t)
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
        print("A11:Count")
        if (len(self.answer) == 1 or type(self.answer) == type([])):
            return len(self.answer.values()[0])
        else:
            print("more than one items in answer")
            return len(self.answer.keys())
        pass

    def at_least(self, N):
        print("A12: at_least")
        answer_keys = []
        for k, v in self.answer.iteritems():
            if len(v) >= N:
                answer_keys.append(k)
        return answer_keys

    def at_most(self, N):
        print("A12: at_most")
        answer_keys = []
        for k, v in self.answer.iteritems():
            if len(v) <= N:
                answer_keys.append(k)
        return answer_keys

    def equal(self, N):
        print("A13: equal")
        answer_keys = []
        for k, v in self.answer.iteritems():
            if len(v) == N:
                answer_keys.append(k)
        return answer_keys

    def EOQ(self):
        pass


    ########################
    def print_answer(self):
        print("----------------")
        if(type(self.answer) == dict):
            for k,v in self.answer.iteritems():
                print self.item_data[k],": ",
                for value in v:
                    print self.item_data[value], ",",
                print
        elif(type(self.answer) == type([])):
            for a in self.answer:
                print self.item_data[a],
            print
        else:
            if(self.answer in self.item_data):
                print self.answer,self.item_data[self.answer]
            else:
                print self.answer
    print("----------------")

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

# -*- coding: utf-8 -*-
# @Time    : 2019/1/18 14:52

from urllib import urlencode

import pickle
import requests
def get_id(idx):
    return int(idx[1:])

class Symbolics():

    def __init__(self, seq, mode='online'):
        if mode != 'online':
            print("loading knowledge base...")
            self.graph = pickle.load(open('/data/zjy/wikidata.pkl', 'rb'))
            self.type_dict = pickle.load(open('/data/zjy/type_kb.pkl', 'rb'))
            print("Load done!")
        else:
            self.graph = None
            self.type_dict = None

        self.seq = seq
        self.answer = {}

    def executor(self):
        for symbolic in self.seq:
            key = symbolic.keys()[0]
            e = symbolic[key][0].strip()
            r = symbolic[key][1].strip()
            t = symbolic[key][2].strip()
            if ("A1" in symbolic):
                self.answer = self.select(e, r.encode("utf-8"), t.encode("utf-8"))
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
                self.answer = self.count(e)
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
            elif ("A15" in symbolic):
                self.answer = self.around(e)
                self.print_answer()
            else:
                print("wrong symbolic")

        return self.answer

    def is_A(self,e):
        #return type of entity
        if self.type_dict is not None:
            try:
                return self.type_dict[get_id(e)]
            except:
                return "empty"
        else:
            json_pack = dict()
            json_pack['op']="is_A"
            json_pack['entity']=e
            content=requests.post("http://127.0.0.1:5000/post",json=json_pack).json()['content']
            return content

    def select(self, e, r, t):
        if self.graph is not None:
            if 'sub' in self.graph[get_id(e)] and r in self.graph[get_id(e)]['sub']:
                return {e:[ee for ee in self.graph[get_id(e)]['sub'][r] if self.is_A(ee) == t]}
            elif 'obj' in self.graph[get_id(e)] and r in self.graph[get_id(e)]['obj']:
                return {e:[ee for ee in self.graph[get_id(e)]['obj'][r] if self.is_A(ee) == t]}
            else:
                return None
        else:
            json_pack = dict()
            json_pack['op'] = "select"
            json_pack['sub'] = e
            json_pack['pre'] = r
            json_pack['obj'] = t
            content = requests.post("http://127.0.0.1:5000/post", json=json_pack).json()['content']
            if content is not None:
                content = set(content)
            return {e:content}

    def select_all(self, et, r, t):
        #print("A2:", et, r, t)
        content = self.answer
        if self.graph is not None and self.par_dict is not None:
            keys = self.par_dict[get_id(et)]
            for key in keys:
                if 'sub' in self.graph[get_id(key)] and r in self.graph[get_id(key)]['sub']:
                    content[key] = [ee for ee in self.graph[get_id(key)]['sub'][r] if self.is_A(ee) == t]
                elif 'obj' in self.graph[get_id(key)] and r in self.graph[get_id(key)]['obj']:
                    content[key] = [ee for ee in self.graph[get_id(key)]['obj'][r] if self.is_A(ee) == t]

                else:
                    content[key] = None
        else:
            json_pack = dict()
            json_pack['op'] = "select_All"
            json_pack['sub'] = et
            json_pack['pre'] = r
            json_pack['obj'] = t

        content = requests.post("http://127.0.0.1:5000/post", json=json_pack).json()['content']
        if self.answer:
            for k in content:
                if k in self.answer:content[k].extend(self.answer[k])
        return content

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
        minK = min(self.answer, key=lambda x: len(self.answer[x]))
        minN = len(self.answer[minK])
        return [k for k in self.answer if len(self.answer[k]) == minN]

    def arg_max(self):
        print("A5: arg_max")
        if not self.answer:
            return None
        maxK = max(self.answer, key=lambda x: len(self.answer[x]))
        maxN = len(self.answer[maxK])
        return [k for k in self.answer if len(self.answer[k]) == maxN]

    def less_than(self, e):
        pass

    def greater_than(self, e):
        pass

    def union(self, e, r, t):
        #print("A9:", e, r, t)
        answer_dict = self.answer
        if e in answer_dict:
            answer_dict[e] = answer_dict[e] | self.select(e, r, t)[e]
        else:
            answer_dict.update(self.select(e, r, t))

        # 进行 union 操作 todo 这里前面都和select部分一样 所以还是应该拆开？ union单独做 好处是union可以不止合并两个 字典里的都可以合并
        union_key = ""
        union_value = set([])
        for k, v in answer_dict.iteritems():
            union_key += k + "|"
            union_value = union_value | set(v)
        union_key = union_key[:-1]
        answer_dict.clear()
        answer_dict[union_key] = set(union_value)

        return answer_dict

    def inter(self, e, r, t):
        #print("A8:", e, r, t)
        answer_dict = self.answer
        if e in answer_dict:
            answer_dict = answer_dict[e] & self.select(e, r, t)[e]
        else:
            answer_dict.update(self.select(e, r, t))

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
        if e in answer_dict:
            answer_dict = answer_dict[e] - self.select(e, r, t)[e]
        else:
            answer_dict.update(self.select(e, r, t))
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

    def count(self,e= None):
        print("A11:Count")
        if type(self.answer) == type([]):
            return len(self.answer)
        elif e!='' and e:
            if e not in self.answer and len(self.answer.keys()) == 1:
                return len(self.answer.popitem())
            return len(self.answer[e])
        else:
            return len(self.answer.keys())

    def at_least(self, N):
        print("A12: at_least")
        # for k in list(self.answer):
        #     if len(self.answer[k]) <= int(N):
        #         self.answer.pop(k)
        # return self.answer
        answer_keys = []
        for k, v in self.answer.iteritems():
            if len(v) >= int(N):
                answer_keys.append(k)
        return answer_keys

    def at_most(self, N):
        print("A12: at_most")
        answer_keys = []
        for k in list(self.answer):
            if len(self.answer[k]) >= int(N):
                self.answer.pop(k)
        return self.answer

    def equal(self, N):
        answer_keys = []
        for k, v in self.answer.iteritems():
            #print k,len(v)
            if len(v) == int(N):
                answer_keys.append(k)
        return answer_keys

    def around(self,N):
        answer_keys = []
        for k, v in self.answer.iteritems():
            #print k, len(v)
            if abs(len(v)-int(N))<(int(N)/2):
                answer_keys.append(k)
        return answer_keys

    def EOQ(self):
        pass

    ########################
    def print_answer(self):
        pass
        # if(type(self.answer) == dict):
        #     for k,v in self.answer.iteritems():
        #         #print self.item_data[k],": ",
        #         for value in v:
        #         #    print self.item_data[value], ",",
        #         print
        # elif(type(self.answer) == type([])):
        #     for a in self.answer:
        #         print self.item_data[a],
        #     print
        # else:
        #     if(self.answer in self.item_data):
        #         print self.answer,self.item_data[self.answer]
        #     else:
        #         print self.answer
    # print("----------------")

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

if __name__ == "__main__":
    print("Building knowledge base....")
    kb = Symbolics(None,'online')
    # for e in kb.find('Q2619632', 'P138'):
    #     print(e,kb.is_A(e))
    for e in kb.select('Q2619632', 'P138', 'Q355304'):
        val = kb.select('Q2619632', 'P138', 'Q355304')[e]
        for v in val:
            print(v, kb.is_A(v))
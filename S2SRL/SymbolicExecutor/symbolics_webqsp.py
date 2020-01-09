# -*- coding: utf-8 -*-
import json
import re
try:
    from urllib import urlencode
except ImportError:
    from urllib.parse import urlencode
import requests
def get_id(idx):
    return int(idx[1:])
from flask import Flask, request, jsonify
app = Flask(__name__)
# Remote Server
# post_url = "http://10.201.34.3:5002/post"
# # local server
post_url = "http://127.0.0.1:5001/post"

class WebQSP(object):
    def __init__(self, id, question, action_sequence_list, entity, relation, type, entity_mask, relation_mask, type_mask, mask_action_sequence_list, answerlist):
        self.id = id
        self.question = question
        self.action_sequence_list = action_sequence_list
        self.entity = entity
        self.relation = relation
        self.type = type
        self.entity_mask = entity_mask
        self.relation_mask = relation_mask
        self.type_mask = type_mask
        self.mask_action_sequence_list = mask_action_sequence_list
        self.answerlist = answerlist

    def obj_2_json(obj):
        return {
            obj.id: {
                "question": obj.question,
                "action_sequence_list": obj.action_sequence_list,
                "entity": obj.entity,
                "relation": obj.relation,
                "type": obj.type,
                "entity_mask": obj.entity_mask,
                "relation_mask": obj.relation_mask,
                "type_mask": obj.type_mask,
                "mask_action_sequence_list": obj.mask_action_sequence_list,
                "answers": obj.answerlist,
            }
        }

class QapairSeq(object):
    def __init__(self, id, question, answer, sparql, seq):
        self.id = id
        self.question = question
        self.answer = answer
        self.sparql = sparql
        self.seq = seq

    def obj_2_json_seq(obj):
        return {
            obj.id:{
                "question": obj.question,
                "answer": obj.answer,
                "sparql": obj.sparql,
                "seq": obj.seq,
            }
        }

class Qapair(object):
    def __init__(self, question, answer, sparql):
        self.question = question
        self.answer = answer
        self.sparql = sparql

    def obj_2_json(obj):
        return {
            "question": obj.question,
            "answer": obj.answer,
            "sparql": obj.sparql,
        }

class Symbolics_WebQSP():

    def __init__(self, seq, mode='online'):
        # local server
        if mode != 'online':
            print("loading local knowledge base...")
            self.freebase_kb = json.load(
                open('../../data/webquestionssp/webQSP_freebase_subgraph.json'))
            print("Loading knowledge is done, start the server...")
            app.run(host='127.0.0.1', port=5001, use_debugger=True)
        # remote server
        else:
            self.graph = None
            self.type_dict = None
        self.seq = seq
        self.answer = {}
        self.temp_variable_list = []    # to store temp variable
        self.temp_set = set([])
        self.temp_bool_dict = {}

    def executor(self):
        if len(self.seq) > 0:
            for symbolic in self.seq:
                # print("current answer:", self.answer)
                key = list(symbolic.keys())[0]
                if len(symbolic[key]) != 3:
                    continue
                e = symbolic[key][0].strip()
                r = symbolic[key][1].strip()
                t = symbolic[key][2].strip()
                # The execution result from A1 is in dict format.
                # A1: Select(e，r，t)
                if ("A1" in symbolic):
                    try:
                        temp_result = self.select_str(e, r, t)
                        self.answer = temp_result
                        self.temp_bool_dict = temp_result
                    except:
                        print('ERROR! The action is Select(%s,%s,%s).' % (e, r, t))
                    finally:
                        self.print_answer()
                elif ("A2" in symbolic):
                    try:
                        temp_result = self.select_e_str(e, r, t)
                        self.answer = temp_result
                        self.temp_bool_dict = temp_result
                    except:
                        print('ERROR! The action is Select(%s,%s,%s).' %(e,r,t))
                    finally:
                        self.print_answer()
                # A3: filter answer
                elif ("A3" in symbolic):
                    try:
                        # ?x ns:a ns:b
                        self.answer.update(self.filter_answer(e, r, t))
                    except:
                        print('ERROR! The action is filter_answer(%s,%s,%s).' % (e, r, t))
                    finally:
                        self.print_answer()
                # A4: Joint
                elif ("A4" in symbolic):
                    try:
                        self.answer.update(self.joint_str(e, r, t))
                    except:
                        print('ERROR! The action is joint_str(%s,%s,%s).' % (e, r, t))
                    finally:
                        self.print_answer()
                # A5: Filter not equal
                elif ("A5" in symbolic):
                    try:
                        self.answer = self.filter_not_equal(e, r, t)
                    except:
                        print('ERROR! The action is filter_not_equal(%s,%s,%s).' % (e, r, t))
                    finally:
                        self.print_answer()
                # # A6: GreaterThan(e)
                # elif ("A6" in symbolic):
                #     try:
                #         self.answer = self.greater_than(e,r,t)
                #     except:
                #         print('ERROR! The action is GreaterThan(%s,%s,%s).' % (e,r,t))
                #     finally:
                #         self.print_answer()
                # # A7: LessThan(e)
                # elif ("A7" in symbolic):
                #     try:
                #         self.answer = self.less_than(e,r,t)
                #     except:
                #         print('ERROR! The action is LessThan(%s,%s,%s).' % (e,r,t))
                #     finally:
                #         self.print_answer()
                # # A9: Union(e，r，t)
                # elif ("A9" in symbolic):
                #     try:
                #         self.answer = self.union(e, r, t)
                #     except:
                #         print('ERROR! The action is Union(%s,%s,%s).' % (e,r,t))
                #     finally:
                #         self.print_answer()
                # # A8: Inter(e，r，t)
                # elif ("A8" in symbolic):
                #     try:
                #         self.answer = self.inter(e, r, t)
                #     except:
                #         print('ERROR! The action is Inter(%s,%s,%s).' % (e,r,t))
                #     finally:
                #         self.print_answer()
                # # A10: Diff(e，r，t)
                # elif ("A10" in symbolic):
                #     try:
                #         self.answer = self.diff(e, r, t)
                #     except:
                #         print('ERROR! The action is Diff(%s,%s,%s).' % (e,r,t))
                #     finally:
                #         self.print_answer()
                # # A11: Count(e)
                # elif ("A11" in symbolic):
                #     try:
                #         self.answer = self.count(e)
                #     except:
                #         print('ERROR! The action is Count(%s).' %e)
                #     finally:
                #         self.print_answer()
                # # A12: ATLEAST(N)
                # elif ("A12" in symbolic):
                #     try:
                #         self.answer = self.at_least(e)
                #     except:
                #         print('ERROR! The action is ATLEAST(%s).' %e)
                #     finally:
                #         self.print_answer()
                # # A13: ATMOST(N)
                # elif ("A13" in symbolic):
                #     try:
                #         self.answer = self.at_most(e)
                #     except:
                #         print('ERROR! The action is ATMOST(%s).' %e)
                #     finally:
                #         self.print_answer()
                # # A14: EQUAL(N)
                # elif ("A14" in symbolic):
                #     try:
                #         self.answer = self.equal(e)
                #     except:
                #         print('ERROR! The action is EQUAL(%s).' %e)
                #     finally:
                #         self.print_answer()
                #
                # elif ("A18" in symbolic):
                #     try:
                #         self.answer = self.joint(e, r, t)
                #     except:
                #         print('ERROR! The action is Joint(%s,%s,%s).' %(e,r,t))
                #     finally:
                #         self.print_answer()
                # elif ("A18_2" in symbolic):
                #     try:
                #         self.answer.update(self.joint_str(e, r, t))
                #     except:
                #         print('ERROR! The action is joint_str(%s,%s,%s).' %(e,r,t))
                #     finally:
                #         self.print_answer()
                # elif ("A19" in symbolic):
                #     try:
                #         # ?x ns:a ns:b
                #         self.answer.update(self.filter_answer(e, r, t))
                #     except:
                #         print('ERROR! The action is filter_answer(%s,%s,%s).' %(e,r,t))
                #     finally:
                #         self.print_answer()
                # elif ("A20" in symbolic):
                #     try:
                #         self.answer = self.filter_not_equal(e, r, t)
                #     except:
                #         print('ERROR! The action is filter_not_equal(%s,%s,%s).' % (e, r, t))
                #     finally:
                #         self.print_answer()
                else:
                    print("wrong symbolic")
        return self.answer

    ########################
    def print_answer(self):
        pass
        # if(type(self.answer) == dict):
        #     for k,v in self.answer.items():
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

    # TODO: NOT TESTED
    # get type
    def is_A(self,e):
        #return type of entity
        if e == "":
            return "empty"
        if self.type_dict is not None:
            try:
                return self.type_dict[get_id(e)]
            except:
                return "empty"
        else:
            json_pack = dict()
            json_pack['op']="is_A"
            json_pack['entity']=e
            content = "empty"
            try:
                # content=requests.post("http://127.0.0.1:5000/post",json=json_pack).json()['content']
                content_json = requests.post("http://10.201.34.3:5000/post", json=json_pack).json()
                if 'content' in content_json:
                    content = content_json['content']
            except:
                print("ERROR for command: is_A(%s)" %e)
            finally:
                return content

    # TODO: NOT THROUGHLY TESTED
    def select(self, e, r, t):
        if e == "" or r == "" or t == "":
            return {}
        else:
            content = set([])
            try:
                json_pack = dict()
                json_pack['op'] = "execute_gen_set1"
                json_pack['sub_pre'] = [e, r]
                jsonpost = json.dumps(json_pack)
                # result_content = requests.post(post_url,json=json_pack)
                # print(result_content)
                content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                if content is not None and content_result==0:
                    content = set(content)
            except:
                print("ERROR for command: select_old(%s,%s,%s)" % (e, r, t))
            finally:
                if content is not None:
                    # Store records in set.
                    content = set(content)
                else:
                    content = set([])
                # A dict is returned whose key is the subject and whose value is set of entities.
                return {t:content}

    def find(self, e, r):
        json_pack = dict()
        json_pack['op'] = "execute_gen_set1"
        json_pack['sub_pre'] = [e, r]
        jsonpost = json.dumps(json_pack)
        content = requests.post(post_url, json=jsonpost).json()['content'][0]
        content_result = requests.post(post_url, json=jsonpost).json()['content'][1]
        if content is not None:
            content = set(content)
        return content

    def select_str(self, e, r, t):
        if e == "" or r == "" or t == "":
            return {}
        else:
            content = set([])
            try:
                json_pack = dict()
                json_pack['op'] = "execute_gen_set1"
                json_pack['sub_pre'] = [e, r]
                jsonpost = json.dumps(json_pack)
                # result_content = requests.post(post_url,json=json_pack)
                # print(result_content)
                #print(jsonpost)
                content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                if content is not None and content_result == 0:
                    content = set(content)
            except:
                print("ERROR for command: select_str(%s,%s,%s)" % (e, r, t))
            finally:
                if content is not None:
                    # Store records in set.
                    content = set(content)
                else:
                    content = set([])
                # A dict is returned whose key is the subject and whose value is set of entities.
                return {t: content}

    def select_e_str(self, e, r, t):
        if e == "" or r == "" or t == "":
            return {}
        else:
            content = set([])
            try:
                json_pack = dict()
                json_pack['op'] = "execute_gen_e_set1"
                json_pack['pre_type'] = [r, t]
                jsonpost = json.dumps(json_pack)
                # result_content = requests.post(post_url,json=json_pack)
                # print(result_content)
                content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                if content is not None and content_result == 0:
                    content = set(content)
            except Exception as error:
                print("ERROR for command: select_e_str(%s,%s,%s)" % (e, r, t), error)
            finally:
                if content is not None:
                    # Store records in set.
                    content = set(content)
                else:
                    content = set([])
                # A dict is returned whose key is the subject and whose value is set of entities.
                return {e: content}


    def select_max_as(self, e, r, t):
        if e == "" or t == "" or r != "":
            return {}
        max = -1
        for item in self.answer[e]:
            item_count = len(self.answer[item])
            if item_count > max:
                max = item_count
        return {t: set(max)}

    # TODO: NOT THROUGHLY TESTED
    # TODO: EXCEPTION HANDLE
    def joint(self, e, r, t):
        intermediate_result = {}
        if e == "" or r == "" or t == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        elif 'VARIABLE' not in self.answer:
            return {}
        else:
            try:
                if e == 'VARIABLE' and t!= 'ANSWER':
                    json_pack = dict()
                    json_pack['op'] = "joint"
                    json_pack['e'] = list(self.answer['VARIABLE'])
                    json_pack['r'] = r
                    json_pack['t'] = t
                    jsonpost = json.dumps(json_pack)
                    # result_content = requests.post(post_url,json=json_pack)
                    # print(result_content)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {'VARIABLE': content}
                if e == 'VARIABLE' and t == 'ANSWER':
                    # print('VARIABLE', self.answer['VARIABLE'])
                    json_pack = dict()
                    json_pack['op'] = "get_joint_answer"
                    json_pack['e'] = list(self.answer['VARIABLE'])
                    json_pack['r'] = r
                    jsonpost = json.dumps(json_pack)
                    # result_content = requests.post(post_url,json=json_pack)
                    # print(result_content)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {'ANSWER': content}
            except:
                print("ERROR for command: joint(%s,%s,%s)" % (e, r, t))
            finally:
                return intermediate_result

    def joint_str(self, e, r, t):
        # print(self.answer)
        intermediate_result = {}
        if e == "" or r == "" or t == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                if '?' in e and t != '?x':
                    json_pack = dict()
                    json_pack['op'] = "joint"
                    json_pack['e'] = list(self.answer[e])
                    json_pack['r'] = r
                    json_pack['t'] = t
                    jsonpost = json.dumps(json_pack)
                    # result_content = requests.post(post_url,json=json_pack)
                    # print(result_content)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {t: content}
                if '?' in e and t == '?x':
                    # print(e, self.answer[e])
                    json_pack = dict()
                    json_pack['op'] = "get_joint_answer"
                    json_pack['e'] = list(self.answer[e])
                    json_pack['r'] = r
                    jsonpost = json.dumps(json_pack)
                    # result_content = requests.post(post_url,json=json_pack)
                    # print(result_content)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {t: content}
            except :
                print("ERROR for command: joint_str(%s,%s,%s)" % (e, r, t))
            finally:
                return intermediate_result

    def filter_answer(self, e, r, t):
        intermediate_result = {}
        if e == "" or r == "" or t == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                # print ("start filter_answer")
                if "?" in e:
                    json_pack = dict()
                    json_pack['op'] = "get_filter_answer"
                    json_pack['e'] = list(self.answer[e])
                    json_pack['r'] = r
                    json_pack['t'] = t
                    jsonpost = json.dumps(json_pack)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    # print(content)
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {e: content}
            except:
                print("ERROR for command: filter_answer(%s,%s,%s)" % (e, r, t))
            finally:
                return intermediate_result

    def filter_not_equal(self, e, r, t):
        intermediate_result = {}
        if e == "" or t == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                # print("filter_not_equal")
                if e in self.answer:
                    answer_list = []
                    # print(t)
                    for answer_item in self.answer[e]:
                        if (answer_item != t):
                            answer_list.append(answer_item)
                    intermediate_result = {e : answer_list}
            except:
                print("ERROR for command: filter_not_equal(%s,%s,%s)" % (e, r, t))
            finally:
                return intermediate_result

    def select_all(self, et, r, t):
        #print("A2:", et, r, t)
        content = {}
        if et == "" or r == "" or t == "":
            return content
        if self.graph is not None and self.par_dict is not None:
            keys = self.par_dict[get_id(et)]
            for key in keys:
                if 'sub' in self.graph[get_id(key)] and r in self.graph[get_id(key)]['sub']:
                    content[key] = [ee for ee in self.graph[get_id(key)]['sub'][r] if self.is_A(ee) == t]
                elif 'obj' in self.graph[get_id(key)] and r in self.graph[get_id(key)]['obj']:
                    content[key] = [ee for ee in self.graph[get_id(key)]['obj'][r] if self.is_A(ee) == t]

                else:
                    content[key] = []
            return content
        else:
            json_pack = dict()
            json_pack['op'] = "select_All"
            json_pack['sub'] = et
            json_pack['pre'] = r
            json_pack['obj'] = t
            try:
                content_json = requests.post("http://10.201.34.3:5000/post", json=json_pack).json()
                if 'content' in content_json:
                    content = content_json['content']
            except:
                print("ERROR for command: select_all(%s,%s,%s)" %(et,r,t))
            # content = requests.post("http://127.0.0.1:5000/post", json=json_pack).json()['content']
            # for k, v in content.items():
            #   if len(v) == 0: content.pop(k)
            finally:
                if self.answer:
                    for k, v in self.answer.items():
                        # Combine the retrieved entities with existed retrieved entities related to same subject.
                        content.setdefault(k, []).extend(v)
                return content

    def is_bool(self, e):
        # print("A3: is_bool")
        if type(self.answer) == bool: return self.answer
        if self.temp_bool_dict == None: return False
        if type(self.temp_bool_dict) == dict:
            for key in self.temp_bool_dict:
                if (self.temp_bool_dict[key] != None and e in self.temp_bool_dict[key]):
                    return True
        return False

    def arg_min(self):
        # print("A4: arg_min")
        if not self.answer:
            return []
        if type(self.answer) != dict: return []
        minK = min(self.answer, key=lambda x: len(self.answer[x]))
        minN = len(self.answer[minK])
        min_set = [k for k in self.answer if len(self.answer[k]) == minN]
        self.temp_set = set(min_set)
        return min_set

    def arg_max(self):
        # print("A5: arg_max")
        if not self.answer:
            return []
        if type(self.answer) != dict: return []
        maxK = max(self.answer, key=lambda x: len(self.answer[x]))
        maxN = len(self.answer[maxK])
        return [k for k in self.answer if len(self.answer[k]) == maxN]

    def greater_than(self, e, r, t):
        content = self.answer
        if type(content) != dict: return []
        if e in content and not content[e] == None:
            N = len(content[e])
        else:
            N = 0
        return [k for k in self.answer if len(self.answer[k]) > N]

    # TODO: NOT TESTED!
    def less_than(self, e, r, t):
        content = self.answer
        if type(content) != dict: return []
        if e in content and not content[e] == None:
            N = len(content[e])
        else:
            N = 0
        return [k for k in self.answer if len(self.answer[k]) < N]

    def union(self, e, r, t):
        #print("A9:", e, r, t)
        if e == "": return {}
        if not e.startswith("Q"): return {}
        answer_dict = self.answer
        if type(answer_dict) == bool: return False
        elif type(answer_dict) != dict: return {}
        try:
            if e in answer_dict and answer_dict[e]!=None:
                temp_dict = self.select(e, r, t)
                if e in temp_dict:
                    answer_dict[e] = set(answer_dict[e]) | set(temp_dict[e])
            else:
                answer_dict.update(self.select(e, r, t))
        except:
            print("ERROR for command: union(%s,%s,%s)" % (e, r, t))
        finally:
            # 进行 union 操作
            # todo 这里前面都和select部分一样 所以还是应该拆开？ union单独做 好处是union可以不止合并两个 字典里的都可以合并
            union_key = "|"
            union_value = set([])
            for k, v in answer_dict.items():
                if v == None: v = []
                union_value = union_value | set(v)
            answer_dict.clear()
            answer_dict[union_key] = list(set(union_value))
            return answer_dict

    # union set e to set t
    def union2t(self, e, r, t):
        #print("A9:", e, r, t)
        if e == "" or t == "": return {}
        if not e.startswith("Q"): return {}
        answer_dict = self.answer
        if type(answer_dict) == bool: return False
        elif type(answer_dict) != dict: return {}
        try:
            if e in answer_dict and answer_dict[e]!=None:
                temp_dict = self.select(e, r, t)
                if e in temp_dict:
                    answer_dict[e] = set(answer_dict[e]) | set(temp_dict[e])
            else:
                answer_dict.update(self.select(e, r, t))
        except:
            print("ERROR for command: union(%s,%s,%s)" % (e, r, t))
        finally:
            # 进行 union 操作
            # todo 这里前面都和select部分一样 所以还是应该拆开？ union单独做 好处是union可以不止合并两个 字典里的都可以合并
            union_key = "|"
            union_value = set([])
            for k, v in answer_dict.items():
                if v == None: v = []
                union_value = union_value | set(v)
            answer_dict.clear()
            answer_dict[union_key] = list(set(union_value))
            return answer_dict

    # equal, or equal
    def filter_or_equal(self, e, r, t):
        self.answer[e].add(t)
        return self.answer[e]

    def count(self,e= None):
        #print("A11:Count")
        try:
            # list or set
            if type(self.answer) == type([]) or type(self.answer) == type(set()):
                return len(self.answer)
            # dict
            if type(self.answer) == type({}):
                if e!='' and e:
                    if e not in self.answer and len(self.answer.keys()) == 1:
                        return len(self.answer.popitem())
                    elif e in self.answer:
                        return len(self.answer[e])
                else:
                    return len(self.answer.keys())
            # int
            if type(self.answer) == type(1):
                return self.answer
            else:
                return 0
        except:
            print("ERROR! THE ACTION IS count(%s)!" %e)
            return 0

    # TODO: NOT TESTED
    def at_least(self, N):
        # print("A12: at_least")
        # for k in list(self.answer):
        #     if len(self.answer[k]) <= int(N):
        #         self.answer.pop(k)
        # return self.answer
        answer_keys = []
        if type(self.answer) == dict:
            for k, v in self.answer.items():
                if len(v) >= int(N):
                    answer_keys.append(k)
        return answer_keys

    # TODO: NOT TESTED
    def at_most(self, N):
        # print("A13: at_most")
        answer_keys = []
        # for k, v in self.answer.items():
        #   if len(v) == 0: self.answer.pop(k)
        if type(self.answer) == dict:
            for k in list(self.answer):
                if len(self.answer[k]) <= int(N):
                    answer_keys.append(k)
        return answer_keys

    # TODO: NOT TESTED
    def equal(self, N):
        answer_keys = []
        if type(self.answer) == dict:
            for k, v in self.answer.items():
                #print k,len(v)
                if len(v) == int(N):
                    answer_keys.append(k)
        return answer_keys

    def both_a(self, e1, e2, r):
        intermediate_result = {}
        if e1 == "" or e2 == "" or r == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                if e1 == 'ANSWER' and e2 != 'VARIABLE':
                    json_pack = dict()
                    json_pack['op'] = "both_a"
                    json_pack['e1'] = list(self.answer[e1])
                    json_pack['e2'] = list(self.answer[e2])
                    json_pack['r'] = r
                    jsonpost = json.dumps(json_pack)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {'ANSWER': content}
            except:
                print("ERROR for command: both_a(%s,%s,%s)" % (e1, e2, r))
            finally:
                return intermediate_result

    def date_less_or_equal(self, e, date):
        intermediate_result = {}
        if e == "" or date == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                if "?" in e:
                    json_pack = dict()
                    json_pack['op'] = "execute_select_oper_date_lt"
                    json_pack['e'] = list(self.answer[e])
                    json_pack['date'] = date(json_pack)
                    jsonpost = json.dumps(json_pack)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {e: content}
            except:
                print("ERROR for command: date_less_or_equal(%s, %s)" % (e, date))
            finally:
                return intermediate_result

    def date_less_or_equal_str(self, e, date):
        intermediate_result = {}
        if e == "" or date == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                if "?" in e:
                    json_pack = dict()
                    json_pack['op'] = "execute_select_oper_date_lt"
                    json_pack['set_date'] = list(self.answer[e])
                    json_pack['date'] = date(json_pack)
                    jsonpost = json.dumps(json_pack)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {e: content}
            except:
                print("ERROR for command: date_less_or_equal(%s, %s)" % (e, date))
            finally:
                return intermediate_result

    def date_greater_or_equal(self, e, date):
        intermediate_result = {}
        if e == "" or date == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                if e == 'VARIABLE':
                    json_pack = dict()
                    json_pack['op'] = "execute_select_oper_date_gt"
                    json_pack['set_date'] = list(self.answer[e])
                    json_pack['date'] = date
                    jsonpost = json.dumps(json_pack)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {e: content}
            except:
                print("ERROR for command: date_greater_or_equal(%s, %s)" % (e, date))
            finally:
                return intermediate_result

    def date_greater_or_equal_str(self, e, date):
        intermediate_result = {}
        if e == "" or date == "":
            return {}
        elif not isinstance(self.answer, dict):
            return {}
        else:
            try:
                if "?" in e:
                    json_pack = dict()
                    json_pack['op'] = "execute_select_oper_date_gt"
                    json_pack['e'] = list(self.answer[e])
                    json_pack['date'] = date
                    jsonpost = json.dumps(json_pack)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {e: content}
            except:
                print("ERROR for command: date_greater_or_equal(%s, %s)" % (e, date))
            finally:
                return intermediate_result

# action sequence
class Action():
    def __init__(self, action_type, e, r, t):
        self.action_type = action_type
        self.e = e
        self.r = r
        self.t = t

    def to_str(self):
        return "{{\'{0}\':[\'{1}\', \'{2}\', \'{3}\']}}".format(self.action_type, self.e, self.r, self.t)

# parse sparql in dataset to action sequence
def processSparql(sparql_str, id="empty"):
        sparql_list = []
        untreated_list = sparql_str.split("\n")
        answer_keys = []
        for untreated_str in untreated_list:
            action_type = "A1"
            s = ""
            r = ""
            t = ""

            # remove note
            note_index = untreated_str.find("#")
            if note_index != -1:
                untreated_str = untreated_str[0:note_index]
            # remove /t
            untreated_str = untreated_str.replace("\t", "")
            untreated_str = untreated_str.strip()
            if untreated_str == '':
                continue

            if "UNION" == untreated_str:
                print ("has union", id)

            if untreated_str.startswith("SELECT"):  # find answer key
                for item in untreated_str.split(" "):
                    if "?" in item:
                        answer_keys.append(item.replace(" ", ""))
            elif untreated_str.startswith("PREFIX") or "langMatches" in untreated_str:
                # ignore
                pass
            elif untreated_str.count("?") == 1 and untreated_str.startswith("ns:"):
                # base action: select
                action_type = "A1"
                triple_list = untreated_str.split(" ")
                if len(triple_list) == 4:
                    s = triple_list[0].replace("ns:", "")
                    r = triple_list[1].replace("ns:", "")
                    t = triple_list[2].replace("ns:", "")
                    if s != "" and r != "" and t != "":
                        action_item = Action(action_type, s, r, t)
                        if isValidAction(action_item):
                            sparql_list.append(action_item)
            elif untreated_str.startswith("FILTER (?x != ns:"):
                # filter not equal
                action_type = "A5"
                s = "?x"
                # s = "ANSWER"
                t = untreated_str.replace("FILTER (?x != ns:", "").replace(")", "").replace(" ", "")
                action_item = Action(action_type, s, r, t)
                if isValidAction(action_item):
                    sparql_list.append(action_item)
            elif untreated_str.count("?") == 1 and untreated_str.startswith("?"):
                # ?x ns:a ns:b
                # if have e,  A3 : filter variable: find sub set fits the bill
                # if don't have e, A1_3 :find e
                action_type = "A3"
                triple_list = untreated_str.split(" ")
                if True:
                    s = triple_list[0].replace("ns:", "")
                    r = triple_list[1].replace("ns:", "")
                    t = triple_list[2].replace("ns:", "")
                    if s != "" and r != "" and t != "":
                        for action in sparql_list:
                            if action.e == s or action.t == s:  # already has variable
                                action_type = "A4"
                        # special for webqsp: swap s and t ,A3->A1 "6" means single action_seq
                        print(len(untreated_list), "length of untreated_list")
                        if len(untreated_list) == 6:
                            action_type = "A1"
                            temp = s
                            s = t
                            t = temp
                        action_item = Action(action_type, s, r, t)
                        if isValidAction(action_item):
                            sparql_list.append(action_item)
                # print(action_item)
            elif untreated_str.count("?") == 2 and ("FILTER" not in untreated_str or "EXISTS" not in untreated_str):
                action_type = "A4"  # joint
                triple_list = untreated_str.split(" ")
                if len(triple_list) == 4:
                    s = triple_list[0].replace("ns:", "")
                    r = triple_list[1].replace("ns:", "")
                    t = triple_list[2].replace("ns:", "")
                    if s != "" and r != "" and t != "":
                        action_item = Action(action_type, s, r, t)
                        if isValidAction(action_item):
                            sparql_list.append(action_item)
                    # print(untreated_str, action_item.to_str())
            # elif untreated_str.startswith("FILTER(xsd:datetime") and "<=" in untreated_str: # filter datetime less or equal
            #     date_re = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", untreated_str)
            #     date_str = date_re.group(0)
            #
            #     date_variable_re = re.search(r"(xsd:datetime\((.*?)\))", untreated_str)
            #     # date_variable_str = date_variable_re.group(1)
            #     action_type = "A22"
            #     t = date_variable_re.group(0)
            #     action_item = Action(action_type, s, r, t)
            #     sparql_list.append(action_item)
            #     print(date_variable_re)
            # elif untreated_str.startswith("FILTER(xsd:datetime") and ">=" in untreated_str: # filter datetime greater or equal
            #     date_str = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", untreated_str)
            #     action_type = "A23"
            #     t = date_variable_re.group(0)
            #     action_item = Action(action_type, s, r, t)
            #     sparql_list.append(action_item)

        # reorder list
        reorder_sparql_list = reorder(sparql_list, answer_keys)
        # reorder_sparql_list = sparql_list
        # for astr in reorder_sparql_list:
        #     print(astr.to_str())

        old_sqarql_list = []
        for item in reorder_sparql_list:
            seqset = {}
            seqlist = []
            seqlist.append(item.e)
            seqlist.append(item.r)
            seqlist.append(item.t)
            seqset[item.action_type] = seqlist
            old_sqarql_list.append(seqset)
        return old_sqarql_list

def isValidAction(action_item):
    return (action_item.e.startswith("m.") or action_item.e.startswith("?"))\
           and (action_item.t.startswith("m.") or action_item.t.startswith("?"))

# todo still has problem with the order of A1_2 and a19 of one variable
def reorder(sparql_list, answer_keys):
    count = 0
    final_len = len(sparql_list)
    reorder_sparql_list = []

    last_variable = ""
    # while len(sparql_list) != 0:
    for key in answer_keys:
        # has_last_select = False
        add_next_variable(sparql_list, key, reorder_sparql_list)
        # print(reorder_sparql_list)
            # contains key
            # 提取包含key并排序
            # for action_item in sparql_list:
            #     # last select action of answer key
            #     if action_item.t == key:
            #         reorder_sparql_list.append(action_item)
            #         sparql_list.remove(action_item)
            #         break
            #
            #         if action_item.e.startswith('?'):
            #             last_variable = action_item.e
            #         has_last_select = True
            #     if has_last_select:
            #         seq = action_item.to_str()
            #         print (seq)
            #         if len(sparql_list) == 1:
            #             reorder_sparql_list.append(action_item)
            #             sparql_list.remove(action_item)
            #             break
            #         if "?x" in seq:
            #             if seq.count("?") == 1:
            #                 reorder_sparql_list.append(action_item)
            #             elif seq.count("?") == 2:
            #                 reorder_sparql_list.append(action_item)
            #                 # define next variable
            #                 next_variable = action_item.e if (action_item.t == "?x") else action_item.t
            #             sparql_list.remove(action_item)
            #             break
            #         elif next_variable != "" and next_variable in seq:
            #             if seq.count("?") == 1:
            #                 reorder_sparql_list.append(action_item)
            #             elif seq.count("?") == 2:
            #                 reorder_sparql_list.append(action_item)
            #                 # define next variable
            #                 next_variable = action_item.e if (action_item.t == next_variable) else action_item.t
            #             sparql_list.remove(action_item)
            #             break
    reorder_sparql_list.reverse()
    return reorder_sparql_list

def add_next_variable(sparql_list, variable_key, reorder_sparql_list):
    for action_item in sparql_list:
        # filter_not_equal is always the last action
        if action_item.action_type == "A5":
            reorder_sparql_list.append(action_item)
            sparql_list.remove(action_item)
            break

    if variable_key == "":
        return

    variable_sql_list = []
    for sql in sparql_list:
        if variable_key == sql.e or variable_key == sql.t:
            variable_sql_list.append(sql)

    if len(variable_sql_list) == 0:
        return
    if len(variable_sql_list) == 1:
        cur_action = variable_sql_list[0]
        if cur_action == "?x" and "?" not in cur_action.t and cur_action.action_type == "A3":
            cur_action.action_type = "A1"
            e = cur_action.e
            t = cur_action.t
            cur_action.e = t
            cur_action.t = e
            reorder_sparql_list.append(sql)
            sparql_list.remove(sql)
            variable_sql_list.remove(sql)
            return

    next_variable = ""
    for sql in variable_sql_list:
        if sql.action_type == "A3":
            reorder_sparql_list.append(sql)
            sparql_list.remove(sql)
            variable_sql_list.remove(sql)

    for sql in variable_sql_list:
        if sql.action_type == "A4":
            # next_variable
            next_variable = sql.e
            reorder_sparql_list.append(sql)
            sparql_list.remove(sql)
            variable_sql_list.remove(sql)

    for sql in variable_sql_list:
        if sql.action_type == "A2":
            reorder_sparql_list.append(sql)
            sparql_list.remove(sql)
            variable_sql_list.remove(sql)

    for sql in variable_sql_list:
        if sql.action_type == "A1":
            reorder_sparql_list.append(sql)
            sparql_list.remove(sql)
            variable_sql_list.remove(sql)

    return add_next_variable(sparql_list, next_variable, reorder_sparql_list)

w_1 = 0.2
def calc_01_reward(answer, true_answer):
    true_reward = 0.0
    try:
        if len(true_answer) == 0:
            if len(answer) == 0:
                return 1.0
            else:
                return w_1
        else:
            right_count = 0
            for e in answer:
                if e in true_answer:
                    right_count += 1
            return float(right_count) / float(len(true_answer))
    except:
        return true_reward

w_1 = 0.2
def calc_01_reward_type(target_value, gold_entities_set, type = "jaccard"):
    true_reward = 0.0
    if type == "jaccard":
        intersec = set(target_value).intersection(set(gold_entities_set))
        union = set([])
        union.update(target_value)
        union.update(gold_entities_set)
        true_reward = float(len(intersec)) / float(len(gold_entities_set))
    elif type == "recall":
        true_reward = float(len(target_value)) / float(len(gold_entities_set))
    elif type == "f1":
        if len(target_value) == 0:
            prec = 0.0
        else:
            prec = float(len(intersec)) / float(len(target_value))
        rec = float(len(intersec)) / float(len(gold_entities_set))
        if prec == 0 and rec == 0:
            true_reward = 0
        else:
            true_reward = (2.0 * prec * rec) / (prec + rec)
    return true_reward

if __name__ == "__main__":
    print("start symbolics_webqsp")
    # local_sparql = "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.01_2n)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.01_2n ns:tv.tv_program.regular_cast ?y .\n?y ns:tv.regular_tv_appearance.actor ?x .\n?y ns:tv.regular_tv_appearance.character ns:m.015lwh .\n}\n"
    # seq = processSparql(local_sparql)
    # print(seq)
    # symbolic_exe = Symbolics_WebQSP(seq)
    # answer = symbolic_exe.executor()
    # print("answer test1: ", answer)


    # seq74 =[
    #         {'A1': ['m.0fjp3', 'sports.sports_championship.events', '?x']},
    #     ]
    # symbolic_exe = Symbolics_WebQSP(seq74)
    # answer = symbolic_exe.executor()
    # print("answer 74: ", answer)

    seq74 =[
            {'A1': ['m.024v2j', 'people.person.parents', '?x']},
        ]
    symbolic_exe = Symbolics_WebQSP(seq74)
    answer = symbolic_exe.executor()
    print("answer 74: ", answer)

    # Load WebQuestions Semantic Parses
    WebQSPList = []
    WebQSPList_Correct = []
    WebQSPList_Incorrect = []
    json_errorlist = []
    true_count = 0

    errorlist = []

    # errorfile = "errorlist_small.json"
    # with open(errorfile, "r", encoding='UTF-8') as webQaTrain:
    #     load_dictTrain = json.load(webQaTrain)
    #     mytrainquestions = load_dictTrain
    #     print(len(mytrainquestions))
    #     myquestions = mytrainquestions
    #     print(len(myquestions))
    #     for q in myquestions:
    #         question = q["ProcessedQuestion"]
    #         # answer = q["Parses"][0]["Answers"][0]["AnswerArgument"]
    #         Answers = []
    #         id = q["QuestionId"]
    #         answerList = q["Parses"][0]["Answers"]
    #         for an in answerList:
    #             Answers.append(an['AnswerArgument'])
    #         # print(answer)
    #         sparql = q["Parses"][0]["Sparql"]
    #         # print(sparql)
    #         mypair = Qapair(question, Answers, sparql)
    #
    #
    #         if id == "WebQTest-1822":
    #         # if True:
    #             # test seq
    #             test_sparql = mypair.sparql
    #             seq = processSparql(test_sparql)
    #             symbolic_exe = Symbolics_WebQSP(seq)
    #             answer = symbolic_exe.executor()
    #             # print("answer: ", answer)
    #             true_answer = mypair.answer
    #             # print("true_answer: ", true_answer)
    #             # print(type(answer))
    #             try:
    #                 if compare(answer, true_answer):
    #                     # print("correct!")
    #                     true_count += 1
    #                     WebQSPList_Correct.append(mypair)
    #                 else:
    #                     WebQSPList_Incorrect.append(mypair)
    #                     print("answer", answer)
    #                     print("seq", seq)
    #                     print("true_answer", true_answer)
    #                     print("id", id)
    #                     errorlist.append(id)
    #                     json_errorlist.append(q)
    #                     print(" ")
    #                     # print('incorrect!')
    #             except:
    #                 pass
    #                 # print('incorrect!')
    #
    #             WebQSPList.append(mypair)
    #     print("%s correct", true_count)
    #     print (errorlist)
    #     # 写入转换后的json
    #     jsondata = json.dumps(json_errorlist, indent=1)
    #     fileObject = open('errorlist_small_2.json', 'w')
    #     fileObject.write(jsondata)
    #     fileObject.close()

    with open("WebQSP.train.json", "r", encoding='UTF-8') as webQaTrain:
        with open("WebQSP.test.json", "r", encoding='UTF-8') as webQaTest:
            load_dictTrain = json.load(webQaTrain)
            load_dictTest = json.load(webQaTest)
            mytrainquestions = load_dictTrain["Questions"]
            print(len(mytrainquestions))
            mytestquestions = load_dictTest["Questions"]
            print(len(mytestquestions))
            myquestions = mytrainquestions + mytestquestions
            # myquestions = mytrainquestions[0:9] + mytestquestions[0:9]
            print(len(myquestions))

            # total rewards
            total_reward = 0
            test_count = 0
            total_reward_jaccard = 0
            total_reward_precision = 0
            total_reward_recall = 0

            for q in mytrainquestions:
                question = q["ProcessedQuestion"]
                # answer = q["Parses"][0]["Answers"][0]["AnswerArgument"]
                Answers = []
                id = q["QuestionId"]
                answerList = q["Parses"][0]["Answers"]
                for an in answerList:
                    Answers.append(an['AnswerArgument'])
                sparql = q["Parses"][0]["Sparql"]
                mypair = Qapair(question, Answers, sparql)

                # if id == "WebQTest-1822": # test one
                if True: # test all
                    # test seq
                    true_answer = mypair.answer
                    test_sparql = mypair.sparql
                    seq = processSparql(test_sparql, id)
                    symbolic_exe = Symbolics_WebQSP(seq)
                    answer = symbolic_exe.executor()
                    # print("answer: ", answer)
                    # print("true_answer: ", true_answer)
                    try:
                        key = "?x"
                        if key in answer:
                            res_answer = answer[key]
                            reward = calc_01_reward_type(res_answer, true_answer)
                            reward_jaccard = calc_01_reward_type(res_answer, true_answer, "jaccard")
                            reward_recall = calc_01_reward_type(res_answer, true_answer, "recall")
                            reward_precision = calc_01_reward_type(res_answer, true_answer, "precision")
                            test_count += 1
                            if reward == 1.0:
                                # print("correct!")

                                # if get right answer, generate action sequence
                                true_count += 1
                                correct = QapairSeq(id, question, true_answer, sparql, seq)
                                entity = set()
                                relation = set()
                                type = set()
                                e_index = 1
                                r_index = 1
                                t_index = 1
                                for srt in seq:
                                    for k,v in srt.items():
                                        if v[0] != "":
                                            entity.add(v[0])
                                        if v[1] != "":
                                            relation.add(v[1])
                                        if v[2] != "":
                                            type.add(v[2])
                                entity = list(entity)
                                relation = list(relation)
                                type = list(type)
                                entity_mask = dict()
                                relation_mask = dict()
                                type_mask = dict()
                                for e in entity:
                                    dict_entity = {e : "ENTITY{0}".format(e_index)}
                                    entity_mask.update(dict_entity)
                                    e_index += 1
                                for r in relation:
                                    dict_relation = {r : "RELATION{0}".format(r_index)}
                                    relation_mask.update(dict_relation)
                                    r_index += 1
                                for t in type:
                                    dict_type = {t : "TYPE{0}".format(t_index)}
                                    type_mask.update(dict_type)
                                    t_index += 1
                                mask_action_sequence_list = []

                                for srt in seq:
                                    mask_set = {}
                                    masklist = []
                                    a_mask = ""
                                    e_mask = ""
                                    r_mask = ""
                                    t_mask = ""
                                    for k,v in srt.items():
                                        a_mask = k
                                        e_mask_key = v[0]
                                        r_mask_key = v[1]
                                        t_mask_key = v[2]
                                        e_mask = entity_mask[e_mask_key] if e_mask_key != "" else ""
                                        r_mask = relation_mask[r_mask_key] if r_mask_key != "" else ""
                                        t_mask = type_mask[t_mask_key] if t_mask_key != "" else ""
                                    if a_mask != "":
                                        masklist.append(e_mask)
                                        masklist.append(r_mask)
                                        masklist.append(t_mask)
                                        mask_set = {a_mask : masklist}
                                        mask_action_sequence_list.append(mask_set)
                                if id != "" and question != "" and seq != "":
                                    correct_item = WebQSP(id, question, seq, entity, relation, type,entity_mask,
                                                          relation_mask, type_mask, mask_action_sequence_list, answerList)
                                # print(question)
                                # print(answer)
                                WebQSPList_Correct.append(correct_item)
                            else:
                                print('incorrect!', reward)
                                WebQSPList_Incorrect.append(mypair)
                                print("answer", answer)
                                print("seq", seq)
                                print("true_answer", true_answer)
                                print("id", id)
                                errorlist.append(id)
                                json_errorlist.append(q)
                                print(" ")

                            total_reward += reward
                            total_reward_jaccard += reward_jaccard
                            total_reward_recall += reward_recall
                            total_reward_precision += reward_precision


                    except Exception as exception:
                        print(exception)
                        pass

                WebQSPList.append(mypair)

            mean_reward_jaccard = total_reward_jaccard / test_count
            mean_reward_recall = total_reward_recall / test_count
            mean_reward_precision = total_reward_precision / test_count
            print("mean_reward_jaccard: ", mean_reward_jaccard)
            print("mean_reward_recall: ", mean_reward_recall)
            print("mean_reward_precision: ", mean_reward_precision)
            print("{0} pairs correct".format(true_count))
            print(errorlist)

            # # 写入转换后的json
            # jsondata = json.dumps(json_errorlist, indent=1)
            # fileObject = open('errorlist_full.json', 'w')
            # fileObject.write(jsondata)
            # fileObject.close()
            #

            jsondata = json.dumps(WebQSPList_Correct, indent=1, default=WebQSP.obj_2_json)
            fileObject = open('right_answer_reorder_mask.json', 'w')
            fileObject.write(jsondata)
            fileObject.close()

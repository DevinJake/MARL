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
# post_url = "http://10.201.34.3:5001/post"
# # local server
post_url = "http://127.0.0.1:5001/post"

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
                        temp_result = self.select(e, r, t)
                        self.answer = temp_result
                        self.temp_bool_dict = temp_result
                    except:
                        print('ERROR! The action is Select(%s,%s,%s).' %(e,r,t))
                    finally:
                        self.print_answer()
                elif ("A1_2" in symbolic):
                    try:
                        temp_result = self.select_str(e, r, t)
                        self.answer = temp_result
                        self.temp_bool_dict = temp_result
                    except:
                        print('ERROR! The action is Select(%s,%s,%s).' %(e,r,t))
                    finally:
                        self.print_answer()
                # A2: SelectAll (et, r, t)
                elif ("A2" in symbolic or "A16" in symbolic):
                    try:
                        self.answer = self.select_all(e, r, t)
                    except:
                        print('ERROR! The action is SelectAll(%s,%s,%s).' % (e,r,t))
                    finally:
                        self.print_answer()
                # A3: Bool(e)
                elif ("A3" in symbolic):
                    try:
                        bool_temp_result = self.is_bool(e)
                        # dict
                        if type(self.answer) == type({}):
                            if '|BOOL_RESULT|' in self.answer:
                                self.answer['|BOOL_RESULT|'].append(bool_temp_result)
                            else:
                                temp = [bool_temp_result]
                                self.answer.setdefault('|BOOL_RESULT|', temp)
                    except:
                        print('ERROR! The action is Bool(%s).' %e)
                    finally:
                        self.print_answer()
                # A4: ArgMin
                elif ("A4" in symbolic):
                    try:
                        self.answer = self.arg_min()
                    except:
                        print('ERROR! The action is ArgMin.')
                    finally:
                        self.print_answer()
                # A5: ArgMax
                elif ("A5" in symbolic):
                    try:
                        self.answer = self.arg_max()
                    except:
                        print('ERROR! The action is ArgMax.')
                    finally:
                        self.print_answer()
                # A6: GreaterThan(e)
                elif ("A6" in symbolic):
                    try:
                        self.answer = self.greater_than(e,r,t)
                    except:
                        print('ERROR! The action is GreaterThan(%s,%s,%s).' % (e,r,t))
                    finally:
                        self.print_answer()
                # A7: LessThan(e)
                elif ("A7" in symbolic):
                    try:
                        self.answer = self.less_than(e,r,t)
                    except:
                        print('ERROR! The action is LessThan(%s,%s,%s).' % (e,r,t))
                    finally:
                        self.print_answer()
                # A9: Union(e，r，t)
                elif ("A9" in symbolic):
                    try:
                        self.answer = self.union(e, r, t)
                    except:
                        print('ERROR! The action is Union(%s,%s,%s).' % (e,r,t))
                    finally:
                        self.print_answer()
                # A8: Inter(e，r，t)
                elif ("A8" in symbolic):
                    try:
                        self.answer = self.inter(e, r, t)
                    except:
                        print('ERROR! The action is Inter(%s,%s,%s).' % (e,r,t))
                    finally:
                        self.print_answer()
                # A10: Diff(e，r，t)
                elif ("A10" in symbolic):
                    try:
                        self.answer = self.diff(e, r, t)
                    except:
                        print('ERROR! The action is Diff(%s,%s,%s).' % (e,r,t))
                    finally:
                        self.print_answer()
                # A11: Count(e)
                elif ("A11" in symbolic):
                    try:
                        self.answer = self.count(e)
                    except:
                        print('ERROR! The action is Count(%s).' %e)
                    finally:
                        self.print_answer()
                # A12: ATLEAST(N)
                elif ("A12" in symbolic):
                    try:
                        self.answer = self.at_least(e)
                    except:
                        print('ERROR! The action is ATLEAST(%s).' %e)
                    finally:
                        self.print_answer()
                # A13: ATMOST(N)
                elif ("A13" in symbolic):
                    try:
                        self.answer = self.at_most(e)
                    except:
                        print('ERROR! The action is ATMOST(%s).' %e)
                    finally:
                        self.print_answer()
                # A14: EQUAL(N)
                elif ("A14" in symbolic):
                    try:
                        self.answer = self.equal(e)
                    except:
                        print('ERROR! The action is EQUAL(%s).' %e)
                    finally:
                        self.print_answer()

                elif ("A18" in symbolic):
                    try:
                        self.answer = self.joint(e, r, t)
                    except:
                        print('ERROR! The action is Joint(%s,%s,%s).' %(e,r,t))
                    finally:
                        self.print_answer()
                elif ("A18_2" in symbolic):
                    try:
                        self.answer = self.joint_str(e, r, t)
                    except:
                        print('ERROR! The action is joint_str(%s,%s,%s).' %(e,r,t))
                    finally:
                        self.print_answer()
                elif ("A19" in symbolic):
                    try:
                        self.answer = self.filter_answer(e, r, t)
                    except:
                        print('ERROR! The action is filter_answer(%s,%s,%s).' %(e,r,t))
                    finally:
                        self.print_answer()
                elif ("A20" in symbolic):
                    try:
                        self.answer = self.filter_not_equal(e, r, t)
                    except:
                        print('ERROR! The action is filter_not_equal(%s,%s,%s).' % (e, r, t))
                    finally:
                        self.print_answer()
                else:
                    print("wrong symbolic")
        return self.answer

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
                print("ERROR for command: select(%s,%s,%s)" % (e, r, t))
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
                content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                if content is not None and content_result == 0:
                    content = set(content)
            except:
                print("ERROR for command: select(%s,%s,%s)" % (e, r, t))
            finally:
                if content is not None:
                    # Store records in set.
                    content = set(content)
                else:
                    content = set([])
                # A dict is returned whose key is the subject and whose value is set of entities.
                return {t: content}

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
                    print('VARIABLE', self.answer['VARIABLE'])
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
        print(self.answer)
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
                    intermediate_result = {e: content}
                if '?' in e and t == '?x':
                    print(e, self.answer[e])
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
                    intermediate_result = {e: content}
            except:
                print("ERROR for command: joint(%s,%s,%s)" % (e, r, t))
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
                print ("start filter_answer")
                if e == 'ANSWER':
                    json_pack = dict()
                    json_pack['op'] = "filter_answer"
                    json_pack['e'] = list(self.answer['ANSWER'])
                    json_pack['r'] = r
                    json_pack['t'] = t
                    jsonpost = json.dumps(json_pack)
                    content, content_result = requests.post(post_url, json=jsonpost).json()['content']
                    print(content)
                    if content is not None and content_result == 0:
                        content = set(content)
                    else:
                        content = set([])
                    intermediate_result = {'ANSWER': content}
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
                print("filter_not_equal")
                if e == 'ANSWER':
                    answer_list = []
                    print(t)
                    for answer_item in self.answer['ANSWER']:
                        if (answer_item != t):
                            answer_list.append(answer_item)
                    intermediate_result = set(answer_list)
            except:
                print("ERROR for command: filter_answer(%s,%s,%s)" % (e, r, t))
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
def processSparql(sparql_str):
        sparql_list = []
        reorder_sparql_list = []
        untreated_list = sparql_str.split("\n")
        answer_keys = []
        for untreated_str in untreated_list:
            action_type = "A1_2"
            s = ""
            r = ""
            t = ""
            if untreated_str.startswith("SELECT"):  # find answer key
                for item in untreated_str.split(" "):
                    if "?" in item:
                        answer_keys.append(item.replace(" ", ""))
            elif untreated_str.startswith("FILTER (?x != ns:"): # filter not equal
                action_type = "A20"
                s = "ANSWER"
                t = untreated_str.replace("FILTER (?x != ns:", "").replace(")", "").replace(" ", "")
                action_item = Action(action_type, s, r, t)
                sparql_list.append(action_item)
            elif untreated_str.startswith("FILTER (!isLiteral(?x) "):
                pass
            elif untreated_str.count("?") == 1 and ("FILTER" not in untreated_str or "EXISTS" not in untreated_str):
                action_type = "A1_2"
                triple_list = untreated_str.split(" ")
                if len(triple_list) == 4:
                    s = triple_list[0].replace("ns:", "")
                    r = triple_list[1].replace("ns:", "")
                    t = triple_list[2].replace("ns:", "")
                action_item = Action(action_type, s, r, t)
                sparql_list.append(action_item)
            elif untreated_str.count("?") == 2 and ("FILTER" not in untreated_str or "EXISTS" not in untreated_str):
                action_type = "A18_2"
                triple_list = untreated_str.split(" ")
                if len(triple_list) == 4:
                    s = triple_list[0].replace("ns:", "")
                    r = triple_list[1].replace("ns:", "")
                    t = triple_list[2].replace("ns:", "")
                action_item = Action(action_type, s, r, t)
                sparql_list.append(action_item)
            elif untreated_str.startswith("FILTER(xsd:datetime") and "<=" in untreated_str: # filter datetime less or equal
                date_re = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", untreated_str)
                date_str = date_re.group(0)

                date_variable_re = re.search(r"(xsd:datetime\((.*?)\))", untreated_str)
                # date_variable_str = date_variable_re.group(1)
                action_type = "A22"
                t = date_variable_re.group(0)
                action_item = Action(action_type, s, r, t)
                sparql_list.append(action_item)
                print(date_variable_re)
            elif untreated_str.startswith("FILTER(xsd:datetime") and ">=" in untreated_str: # filter datetime greater or equal
                date_str = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", untreated_str)
                action_type = "A23"
                t = date_variable_re.group(0)
                action_item = Action(action_type, s, r, t)
                sparql_list.append(action_item)

        # reorder
        count = 0
        while len(reorder_sparql_list) != len(sparql_list):
            count = count + 1
            if count > len(sparql_list):
                break
            next_variable = ""
            for action_item in sparql_list:
                seq = action_item.to_str()
                if (answer_keys[0]) in seq:
                    if untreated_str.count("?") == 1:
                        reorder_sparql_list.append(seq)
                    elif untreated_str.count("?") == 2:
                        reorder_sparql_list.append(seq)
                        # define next variable
                        next_variable = action_item.e if (action_item.t == answer_keys[0]) else action_item.t
                if next_variable != "" and next_variable in seq:
                    if untreated_str.count("?") == 1:
                        reorder_sparql_list.append(seq)
                    elif untreated_str.count("?") == 2:
                        reorder_sparql_list.append(seq)
                        # define next variable
                        next_variable = action_item.e if (action_item.t == next_variable) else action_item.t

        reorder_sparql_list.reverse()
        return "[" + ",".join(reorder_sparql_list) + "]"

if __name__ == "__main__":
    print("start symbolics_webqsp")
    # # test1
    # symbolic_seq = [{'A1': ['m.09l3p', 'film.actor.film', 'VARIABLE']}, {'A18': ['VARIABLE', 'film.performance.film', 'm.0ddt_']}, {'A18': ['VARIABLE', 'film.performance.character', 'ANSWER']}]
    # symbolic_exe = Symbolics_WebQSP(symbolic_seq)
    # answer = symbolic_exe.executor()
    # print(answer)
    #
    # # test1_2
    # symbolic_seq = [{'A1_2': ['m.09l3p', 'film.actor.film', '?y']}, {'A18_2': ['?y', 'film.performance.film', 'm.0ddt_']}, {'A18_2': ['?y', 'film.performance.character', '?x']}]
    # symbolic_exe = Symbolics_WebQSP(symbolic_seq)
    # answer = symbolic_exe.executor()
    # print("test1_2", answer) # succeed

    # WebQTrn-3               "AnswerArgument": "m.0160w",
    # ns:m.03st9j ns:location.location.containedby ?x .
    # ?x ns:base.biblioness.bibs_location.loc_type ?sk0 .
    # FILTER (str(?sk0) = \"Country\")

    # symbolic_seq = [
    #     {'A1_2': ['m.03st9j', 'location.location.containedby', '?x']},
    #     {'A23': ['?x', 'base.biblioness.bibs_location.loc_type', 'Country']},
    #     # {'A18_2': ['?y', 'film.performance.character', '?x']},
    # ]
    # symbolic_exe = Symbolics_WebQSP(symbolic_seq)
    # answer = symbolic_exe.executor()
    # print("3", answer) # succeed



    # # test2
    # symbolic_seq = [{'A1': ['m.06w2sn5', 'people.person.sibling_s', 'VARIABLE']}, {'A18': ['VARIABLE', 'people.sibling_relationship.sibling', 'ANSWER']}, {'A19': ['ANSWER', 'people.person.gender', 'm.05zppz']}]
    # symbolic_exe = Symbolics_WebQSP(symbolic_seq)
    # answer = symbolic_exe.executor()
    # print(answer)
    #
    # # test3
    # symbolic_seq = [{'A1': ['m.03f2h01', 'base.activism.activist.area_of_activism', 'ANSWER']}]
    # symbolic_exe = Symbolics_WebQSP(symbolic_seq)
    # answer = symbolic_exe.executor()
    # print(answer)
    #
    # #6
    # symbolic_seq = [
    #     {'A1': ['m.0c2yrf', 'sports.pro_athlete.teams', '?y']},
    # ]
    # symbolic_exe = Symbolics_WebQSP(symbolic_seq)
    # answer = symbolic_exe.executor()
    # print(answer)
    #
    # #7
    # symbolic_seq = [
    #     {'A1': ['m.084l5', 'sports.sports_team.location', 'ANSWER']},
    # ]
    # symbolic_exe = Symbolics_WebQSP(symbolic_seq)
    # answer = symbolic_exe.executor()
    # print(7, answer) # y
    #
    # #12
    # symbolic_seq = [
    #     {'A1': ['m.0c2m4', 'fictional_universe.fictional_character.married_to', 'VARIABLE']},
    #     {'A18': ['VARIABLE', 'fictional_universe.marriage_of_fictional_characters.spouses', 'ANSWER']},
    #     {'A20': ['ANSWER', '', 'm.0c2m4']},
    # ]
    # symbolic_exe = Symbolics_WebQSP(symbolic_seq)
    # answer = symbolic_exe.executor()
    # print(12, answer) # 09k254x dui
    #
    # #13
    # symbolic_seq = [
    #     {'A1': ['m.09c7w0', 'location.location.adjoin_s', 'VARIABLE']},
    #     {'A18': ['VARIABLE', 'location.adjoining_relationship.adjoins', 'ANSWER']},
    #     {'A19': ['ANSWER', 'common.topic.notable_types', 'm.01mp']},
    # ]
    # symbolic_exe = Symbolics_WebQSP(symbolic_seq)
    # answer = symbolic_exe.executor()
    # print(13, answer) #    0d060g 错

    #1124
    # "RawQuestion": "who is jennifer lawrence boyfriend 2012?",
    # 06tjh6
    # symbolic_seq = [
    #     {'A1_2': ['m.02x0dzw', 'celebrities.celebrity.sexual_relationships', '?y']},
    #     # {'A18': ['?y', 'celebrities.romantic_relationship.relationship_type', 'm.02_7k44']},
    #     {'A18_2': ['?y', 'celebrities.romantic_relationship.celebrity', '?x']},
    # ]
    # symbolic_exe = Symbolics_WebQSP(symbolic_seq)
    # answer = symbolic_exe.executor()
    # print(1124, answer) #    06tjh6 错 缺失

    #2513


    #21 m.0dl1t3
    variable_set = {}

    # local_sparql = "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.06w2sn5)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.06w2sn5 ns:people.person.sibling_s ?y .\n?y ns:people.sibling_relationship.sibling ?x .\n?x ns:people.person.gender ns:m.05zppz .\n}\n"
    local_sparql = "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.02d9k)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.02d9k ns:sports.pro_athlete.teams ?y .\n?y ns:sports.sports_team_roster.team ?x .\nFILTER(NOT EXISTS {?y ns:sports.sports_team_roster.from ?sk0} || \nEXISTS {?y ns:sports.sports_team_roster.from ?sk1 . \nFILTER(xsd:datetime(?sk1) <= \"2013-12-31\"^^xsd:dateTime) })\nFILTER(NOT EXISTS {?y ns:sports.sports_team_roster.to ?sk2} || \nEXISTS {?y ns:sports.sports_team_roster.to ?sk3 . \nFILTER(xsd:datetime(?sk3) >= \"2013-01-01\"^^xsd:dateTime) })\n}\n"
    seq = processSparql(local_sparql)
    print(seq)

    # WebQSP_data = json.load(open('WebQSP.train.json'))
    # mytrainquestions = WebQSP_data["Questions"]
    # print(len(mytrainquestions))

import os
import requests
from urllib import request
import pickle
from server import Interpreter
import json
def get_id(idx):
    return int(idx[1:])
post_url = "http://10.201.34.3:5001/post"

class KB(object):
    def __init__(self,mode='online'): 
        if mode!='online':
            print("loading knowledge base...")
            self.graph=pickle.load(open('/data/zjy/wikidata.pkl','rb'))
            self.type_dict=pickle.load(open('/data/zjy/type_kb.pkl','rb'))
            self.par_dict=pickle.load(open('/data/zjy/par_dict.pkl','rb'))
            print("Load done!")
        else:
            self.graph=None
            self.type_dict=None
            self.par_dict=None
            
    def find(self,e,r):
        #return find({e},relation)
        if self.graph is not None:
            if 'sub' in self.graph[get_id(e)] and r in self.graph[get_id(e)]['sub']:
                return self.graph[get_id(e)]['sub'][r]
            else:
                return None
        else:
            json_pack = dict() 
            json_pack['op']="find"
            json_pack['sub']=e
            json_pack['pre']=r
            print("start find", e, r)
            jsonpost = json.dumps(json_pack)
            # result_content = requests.post(post_url,json=json_pack)
            # print(result_content)
            content=requests.post(post_url,json=jsonpost).json()['content']
            # content=requests.post(post_url,json=jsonpost)
            print("end find", e, r, content)
            if content is not None:
                content=set(content)            
            return content

    def execute_gen_set1(self, e, r):
        json_pack = dict()
        json_pack['op'] = "execute_gen_set1"
        json_pack['sub_pre'] = [e, r]
        jsonpost = json.dumps(json_pack)
        # result_content = requests.post(post_url,json=json_pack)
        # print(result_content)
        content = requests.post(post_url, json=jsonpost).json()['content'][0]
        content_result = requests.post(post_url, json=jsonpost).json()['content'][1]
        if content is not None:
            content = set(content)
        return content

        
    def find_reverse(self,e,r):
        #return find({e},reverse(relation))
        if self.graph is not None:
            if 'obj' in self.graph[get_id(e)] and r in self.graph[get_id(e)]['obj']:
                return self.graph[get_id(e)]['obj'][r]
            else:
                return None
        else:
            json_pack = dict() 
            json_pack['op']="find_reverse"
            json_pack['obj']=e
            json_pack['pre']=r
            result_content = requests.post(post_url,json=json_pack).json()
            print(result_content)
            content=requests.post(post_url,json=json_pack).json()['content']
            if content is not None:
                content=set(content)
            return content

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
            content=requests.post(post_url,json=json_pack).json()['content']
            return content   

    def is_All(self,t):
        # return entities which type is t.
        if self.par_dict is not None:
            try:
                return self.par_dict[get_id(t)]
            except:
                return "empty"
        else:
            json_pack = dict()
            json_pack['op']="is_All"
            json_pack['type']=t
            content=requests.post(post_url,json=json_pack).json()['content']
            return content

    def select(self,e,r,t):
        # return select(entity,relation,type)
        if self.graph is not None:
            if 'sub' in self.graph[get_id(e)] and r in self.graph[get_id(e)]['sub']:
                return {e:[ee for ee in self.graph[get_id(e)]['sub'][r] if self.is_A(ee) == t]}
            elif 'obj' in self.graph[get_id(e)] and r in self.graph[get_id(e)]['obj']:
                return [ee for ee in self.graph[get_id(e)]['obj'][r] if t in self.is_A(ee)]
            else:
                return None
        else:
            json_pack = dict()
            json_pack['op'] = "select"
            json_pack['sub'] = e
            json_pack['pre'] = r
            json_pack['obj'] = t
            content = requests.post(post_url, json=json_pack).json()['content']
            if content is not None:
                content = set(content)
            return {e:content}

    def select_All(self,et,r,t):
        content = {}
        if self.graph is not None and self.par_dict is not None:
            keys = self.par_dict[get_id(et)]
            for key in keys:
                if 'sub' in self.graph[get_id(key)] and r in self.graph[get_id(key)]['sub']:
                    content[key] = [ee for ee in self.graph[get_id(key)]['sub'][r] if self.is_A(ee) == t]
                elif 'obj' in self.graph[get_id(key)] and r in self.graph[get_id(key)]['obj']:
                    content[key] = [ee for ee in self.graph[get_id(key)]['obj'][r] if self.is_A(ee) == t]

                else:
                    content[key] = None

            pass
        else:
            json_pack = dict()
            json_pack['op'] = "select_All"
            json_pack['sub'] = et
            json_pack['pre'] = r
            json_pack['obj'] = t

        content = requests.post(post_url, json=json_pack).json()['content']

        return content

    def test(self):
        json_pack = dict()
        content = requests.post(post_url, json=json_pack).json()
        return content


if __name__ == "__main__":
    print("Building knowledge base....")
    kb = KB()
    # print(kb.test())

    # result = kb.find("m.011m3hqc", "music.album.release_date")
    # print(result)

    result = kb.execute_gen_set1("m.0121zk7p", "tv.tv_series_episode.air_date")
    result = kb.execute_gen_set1("2015-03-12", "tv.tv_series_episode.air_date")
    print(result)

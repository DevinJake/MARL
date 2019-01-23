# -*- coding: utf-8 -*-
# @Time    : 2019/1/20 16:40

import collections

class Seq2Seq():

    def simple(self,qstring,entities,relations,types,states):

        # 操作序列 (操作符：[参数])
        print("--------------------------------------------------------------------------------")
        print("Question:", qstring)
        print("Entities:", entities,"Relations:", relations,"Types:", types, "State:",states)
        symbolic_seq = []

        if(states == 1):
            print("ChangedQuestion:", "Which rivers flow through India or China?")
            symbolic_seq.append({"A1": ['Q148', 'P17', 'Q4022']}) # select（China, flow, River），Union(India, flow, River), EOQ
            symbolic_seq.append({"A9": ['Q668', 'P17', 'Q4022']})
        elif(states == 2):
            print("ChangedQuestion:", "Which rivers flow through India and China?")
            symbolic_seq.append({"A1": ['Q148', 'P17', 'Q4022']})  # select（China, flow, River），Inter(India, flow, River)，EOQ
            symbolic_seq.append({"A10": ['Q668', 'P17', 'Q4022']})
        elif (states == 3):
            print("ChangedQuestion:", "Which rivers flow through India but not China?")
            symbolic_seq.append({"A1": ['Q148', 'P17', 'Q4022']})  # select（China, flow, River）， Diff(India, flow, River)，EOQ
            symbolic_seq.append({"A11": ['Q668', 'P17', 'Q4022']})
        elif (states == 4):
            print("ChangedQuestion:", "Does Ganga flow through India?")
            symbolic_seq.append({"A1": ['Q148', 'P17', 'Q4022']})  # select(India, flow, River), Bool(Ganga, in)，EOQ
            symbolic_seq.append({"A3": ['Q5413','','']})
        elif (states == 5):
            print("ChangedQuestion:", "Which river flows through India but does not continent in Asia?")
            symbolic_seq.append({"A1": ['Q148', 'P17', 'Q4022']})  # select(India, Flow, River)，Diff(Asia,continent, River)，EOQ
            symbolic_seq.append({"A11": ['Q48', 'P30', 'Q4022']})
        elif (states == 6):
            print("ChangedQuestion:", "How many rivers flow through India ?")
            symbolic_seq.append({"A1": ['Q148', 'P17', 'Q4022']})  # select(India, Flow, River)，Count，EOQ
            symbolic_seq.append({"A12": ['','','']})
        elif (states == 7):
            print("ChangedQuestion:", "How many rivers and lakes does India have ?")
            symbolic_seq.append({"A1": ['Q668', 'P17', 'Q4022']}) # select(India, Flow, River)，Union(India，Flow，Lake), Count，EOQ
            symbolic_seq.append({"A9": ['Q668', 'P17', 'Q23397']})
            symbolic_seq.append({"A12": ['', '', '']})
        elif (states == 8):
            print("ChangedQuestion:", "Which river flows through maximum number of countries ?")
            symbolic_seq.append({"A2": ['Q15617994', 'P17', 'Q4022']}) # select(India, Flow, River)，Union(India，Flow，Lake), Count，EOQ
            symbolic_seq.append({"A12": ['', '', '']})
        else:
            print(states,"not ready")


        #
        # select(River, Flow, River)，Union / Intesrsection / Difference(River，China，Flow)，Count，EOQ

        # entity ["Q148:People's Republic of China"] u'Q5413:Yangtze' Q668 India
        # relation 'P17:country'
        # type[river:Q4022] lake Q23397
        # "P177": "crosses","P469": "lakes on river", "P17": "country" "P127": "owned by","P276": "location",
        # "P47": "shares border with",
        # type[river:Q4022] [lake:Q23397] [country:"Q6256"]
        return symbolic_seq
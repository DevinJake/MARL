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
            symbolic_seq.append({"A8": ['Q668', 'P17', 'Q4022']})
        elif (states == 3):
            print("ChangedQuestion:", "Which rivers flow through India but not China?")
            symbolic_seq.append({"A1": ['Q148', 'P17', 'Q4022']})  # select（China, flow, River）， Diff(India, flow, River)，EOQ
            symbolic_seq.append({"A10": ['Q668', 'P17', 'Q4022']})
        elif (states == 4):
            print("ChangedQuestion:", "Does Yangtze flow through China?")
            symbolic_seq.append({"A1": ['Q148', 'P17', 'Q4022']})  # select(China, flow, River), Bool(Yangtze)，EOQ
            symbolic_seq.append({"A3": ['Q5413','','']})
        elif (states == 5):
            print("ChangedQuestion:", "Which river flows through India but does not continent in Asia?")
            symbolic_seq.append({"A1": ['Q148', 'P17', 'Q4022']})  # select(India, Flow, River)，Diff(Asia,continent, River)，EOQ
            symbolic_seq.append({"A10": ['Q48', 'P30', 'Q4022']})
        elif (states == 6):
            print("ChangedQuestion:", "How many rivers flow through India ?")
            symbolic_seq.append({"A1": ['Q148', 'P17', 'Q4022']})  # select(India, Flow, River)，Count，EOQ
            symbolic_seq.append({"A11": ['','','']})
        elif (states == 7):
            print("ChangedQuestion:", "How many rivers and lakes does India have ?")
            symbolic_seq.append({"A1": ['Q668', 'P17', 'Q4022']}) # select(India, Flow, River)，Union(India，Flow，Lake), Count，EOQ
            symbolic_seq.append({"A9": ['Q668', 'P17', 'Q23397']})
            symbolic_seq.append({"A11": ['', '', '']})
        elif (states == 8):
            print("ChangedQuestion:", "Which river flows through maximum number of countries ?")
            symbolic_seq.append({"A2": ['Q4022', 'P17', 'Q6256']}) #selectAll(River, Flow, Country),argmax,EOQ
            symbolic_seq.append({"A5": ['', '', '']})
        elif (states == 9):
            print("ChangedQuestion:", "Which country has maximum number of rivers and lakes combined ?")# todo test is not ready, need reverse data
            symbolic_seq.append({"A2": ['Q6256', 'P17', 'Q4022']}) #selectAll(Country, reverse(Flow), River),selectAll(Country, reverse(Flow), Lake),argmax,EOQ
            symbolic_seq.append({"A2": ['Q6256', 'P17', 'Q23397']})
            symbolic_seq.append({"A5": ['', '', '']})
        elif (states == 10):
            print("ChangedQuestion:", "Which rivers flow through at least N countries ?")
            symbolic_seq.append({"A2": ['Q4022', 'P17', 'Q6256']})  # selectAll(River, Flow, Country),at_least_N,EOQ
            symbolic_seq.append({"A12": [2, '', '']})
        elif (states == 11):
            print("ChangedQuestion:", "Which river flows through maximum number of countries ?")
            symbolic_seq.append({"A2": ['Q4022', 'P17', 'Q6256']}) #
            symbolic_seq.append({"A5": ['', '', '']})
        elif (states == 12):
            print("ChangedQuestion:", "Which country has at least N rivers and lakes combined ?")# todo
            symbolic_seq.append({"A2": ['Q6256', 'P17', 'Q4022']})  # selectAll(Country, reverse(Flow), River),selectAll(Country, reverse(Flow), Lake),argmax,EOQ
            symbolic_seq.append({"A2": ['Q6256', 'P17', 'Q23397']})
            symbolic_seq.append({"A12": [2, '', '']})
        elif (states == 13):
            print("ChangedQuestion:", "How many rivers flow through at least N countries?")
            symbolic_seq.append({"A2": ['Q4022', 'P17', 'Q6256']})  #
            symbolic_seq.append({"A12": [2, '', '']})
            symbolic_seq.append({"A11": ['', '', '']})
        elif (states == 14):
            print("ChangedQuestion:", "How many country has at least N rivers and lakes combined ?")# todo
            symbolic_seq.append({"A2": ['Q6256', 'P17', 'Q4022']})  # selectAll(Country, reverse(Flow), River),selectAll(Country, reverse(Flow), Lake),argmax,EOQ
            symbolic_seq.append({"A2": ['Q6256', 'P17', 'Q23397']})
            symbolic_seq.append({"A12": [2, '', '']})
            symbolic_seq.append({"A11": ['', '', '']})
        elif (states == 15):
            print("ChangedQuestion:", "Which countries have more number of rivers than India ?")
            symbolic_seq.append({"A2": ['Q6256', 'P17', 'Q4022']})
            symbolic_seq.append({"A2": ['Q668']})
        elif (states == 16):
            print("ChangedQuestion:", "Which countries have more number of rivers AND lakes than India ?")
            symbolic_seq.append({"A2": ['Q6256', 'P17', 'Q4022']})
            symbolic_seq.append({"A2": ['Q6256', 'P17', 'Q23397']})
            symbolic_seq.append({"A2": ['Q668']})
        elif (states == 17):
            print("ChangedQuestion:", "How many countries have more number of rivers than India ?")
            symbolic_seq.append({"A2": ['Q6256', 'P17', 'Q4022']})
            symbolic_seq.append({"A2": ['Q668']})
            symbolic_seq.append({"A11": ['', '', '']})
        elif (states == 18):
            print("ChangedQuestion:", "How many countries have more number of rivers and lakes than India ?")
            symbolic_seq.append({"A2": ['Q6256', 'P17', 'Q4022']})
            symbolic_seq.append({"A2": ['Q6256', 'P17', 'Q23397']})
            symbolic_seq.append({"A2": ['Q668']})
            symbolic_seq.append({"A11": ['', '', '']})
        else:
            print(states,"not ready")


        # entity      ["Q148:People's Republic of China"]  u'Q5413:Yangtze'  Q668 India
        # relation    'P17:country'
        # type        [river:Q4022] [lake:Q23397] [country:"Q6256"]
        return symbolic_seq
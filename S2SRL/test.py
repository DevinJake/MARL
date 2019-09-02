# -*- coding: utf-8 -*-
# @Time    : 2019/09/02 20:25
# @Author  : Devin Hua

'''
input:
<class 'list'>: ['A2', '(', 'Q6619679', 'P674', 'Q838948', ')', 'A2', '(', 'Q6619679', '-', 'P674', 'Q36649', ')', 'A6', '(', 'Q944203', ')', 'A11', '(', ')']
output:
<class 'list'>: [{'A2': ['Q3895768', 'P1441', 'Q3895768']}, {'A2': ['Q15961987', '-P1441', 'Q95074']}, {'A15': ['Q347947', '', '']}]
'''
from SymbolicExecutor.transform_util import list2dict

if __name__ == "__main__":
    list_temp = ['A2', '(', ')', ')', '(', 'Q6619679', '-', 'P674', 'Q36649', 'A6', 'Q944203']
    print(list2dict(list_temp))
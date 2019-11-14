#-*-coding:utf-8-*-
"""
This file is used to test certain functions or methods.
"""
import random
import os
import numpy as np
import torch

def transformBooleanToString(list):
    temp_set = set()
    if len(list) == 0:
        return ''
    else:
        for i, item in enumerate(list):
            if item == True:
                list[i] = "YES"
                temp_set.add(list[i])
            elif item == False:
                list[i] = "NO"
                temp_set.add(list[i])
            else:
                return ''
    if len(temp_set) == 1:
        return temp_set.pop()
    if len(temp_set) > 1:
        return ((' and '.join(list)).strip() + ' respectively')

def test():
    a = [2 - 1.5] * 10
    print (a)

    actions_t = slice(3)
    log_prob_v = [[1,2,3],[4,5,6],[7,8,9]]
    a = log_prob_v[actions_t]
    print(a)

    answer = 111
    orig_response = 222
    print (str(answer) + "::" + str(orig_response))
    print ("%s::%s" %(answer, orig_response))
    temp_string = "%s::%s" %(answer, orig_response)
    print (temp_string)

    line = '''symbolic_seq.append({"A1": ['Q910670', 'P47', 'Q20667921']})'''
    key = line[line.find("{") + 1:line.find('}')].split(':')[0].replace('\"', '').strip()
    print (key)
    val = line[line.find("{") + 1: line.find('}')].split(':')[1].strip()
    print (val)
    val = val.replace('[', '')
    print(val)
    val = val.replace(']', '')
    print(val)
    val = val.replace("\'", "")
    print(val)
    val = val.strip()
    print(val)
    val =val.split(',')
    print(val)

    temp_list = list()
    temp_dict = {"a":1}
    if 'a' in temp_dict:
        print ("a")
    if '1' in temp_dict:
        print("1")
    temp_list.append(temp_dict)
    temp_list.append(False)
    print (temp_list)
    for item in temp_list:
        print (item.__class__)

    print (transformBooleanToString([True, False, True]))
    print (transformBooleanToString([True, False]))
    print (transformBooleanToString([True, True]))
    print (transformBooleanToString([False, False]))
    print (transformBooleanToString([False, False, True]))
    print (transformBooleanToString([False, False, False]))
    print (transformBooleanToString([False, False, 'DEVIN']))
    print (transformBooleanToString([]))

    n = '16'
    print (n.isdigit())
    n = 'Q12416'
    print (n.isalnum())

    number = 1
    if number == 0 or 1 < number <= 5:
        print (True)
    else:
        print (False)

    context_entities = "Q910670|Q205576|Q334"
    context_entities = [x.strip() for x in context_entities.split('|')]
    context = "In Q333 does Q910670 share the border with Q205576 ?"
    entity_index = {x : (context.find(x) if context.find(x)!=-1 else 100000) for x in context_entities}
    entity_index = sorted(entity_index.items(), key = lambda item:item[1])
    temp_string = ','.join([x[0].strip() for x in entity_index])
    print (entity_index)
    print (temp_string)

    d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    print(d)
    l = list(d.items())
    random.shuffle(l)
    d = dict(l)
    print (d)

    print (type(d) == type({}))

    s = "<E> ENTITY1 </E> <R> RELATION1 </R> <T> TYPE1 TYPE2 </T> which"
    print(len(s))

    p = [(1, 2), (3, 4), (5, 6), (7, 8)]
    d = zip(*p)
    print(list(d))
    print (os.path.abspath(os.path.join(os.getcwd(), "../..")))


# !/usr/bin/env python

# This is a straightforward implementation of a well-known algorithm, and thus
# probably shouldn't be covered by copyright to begin with. But in case it is,
# the author (Magnus Lie Hetland) has, to the extent possible under law,
# dedicated all copyright and related and neighboring rights to this software
# to the public domain worldwide, by distributing it under the CC0 license,
# version 1.0. This software is distributed without any warranty. For more
# information, see <http://creativecommons.org/publicdomain/zero/1.0>

def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def duplicate(s1,s2):
    compare = lambda a,b: len(a)==len(b) and len(a)==sum([1 for i,j in zip(a,b) if i==j])
    return compare(s1, s2)

if __name__ == "__main__":
    test()
    s = '15'
    if s.isdigit():
        print(int(s))
    for i in range(2-1):
        print ('hua')
    a = [float(i/2) for i in range(4)]
    print(np.mean(a))
    print(float(np.mean(a)))

    answer = {}
    if type(answer) == dict:
        temp = []
        for key, value in answer.items():
            if key != '|BOOL_RESULT|' and value:
                temp.extend(list(value))
        predicted_answer = temp
    print(predicted_answer)
    diff_value = set()
    temp_set = {1,2,3}
    diff_value = diff_value - temp_set
    for temp in diff_value:
        print(temp)
    a = {1,2,3}
    if(type(a)== type(set())):
        print(len(a))
    print(type({}))
    print(type(set()))
    answer = 1
    if type(answer) == bool: print(answer)
    s1 = [17,11,47,23,5]
    s2 = [11,47,23,5,17]
    print(duplicate(s1,s2))

    d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    for k in list(d):
        print(k)

    list2 = [0,1]
    list1 = [1,1]
    similarity = 1.0 - float(levenshtein(list1,list2))/float(max(len(list1),len(list2)))
    print(max(len(list1),len(list2)))
    print(levenshtein(list1,list2))
    print(similarity)

    lengths = np.array([50] * 5)
    eos_batches = np.array([0, 0, 1, 0, 0])
    print(lengths > 1)
    print(((lengths > 1) & eos_batches))
    update_idx = ((lengths > 1) & eos_batches) != 0
    print(update_idx)
    lengths[update_idx] = 2
    print(lengths)

    batch_size = 5
    inputs = torch.LongTensor([0] * batch_size).view(batch_size, 1)
    print(inputs)
    decoder_input = inputs[:, 0].unsqueeze(1)
    print(decoder_input)

    for i in range(10):
        print(random.random())





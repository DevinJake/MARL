# -*- coding: utf-8 -*-
# @Time    : 2019/9/1 23:36
# @Author  : Devin Hua
# Function: transforming.

# Transform boolean results into string format.
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

def list2dict(list):
    final_list = []
    temp_list = []
    new_list = []
    for a in list:
        if (a == "("):
            new_list = []
            continue
        if (a == ")"):
            if ("-" in new_list and new_list[-1] != "-"):
                new_list[new_list.index("-") + 1] = "-" + new_list[new_list.index("-") + 1]
                new_list.remove("-")
            if (new_list == []):
                new_list = ["", "", ""]
            if (len(new_list) == 1):
                new_list = [new_list[0], "", ""]
            if ("&" in new_list):
                new_list = ["&", "", ""]
            if ("-" in new_list):
                new_list = ["-", "", ""]
            if ("|" in new_list):
                new_list = ["|", "", ""]
            temp_list.append(new_list)
            continue
        if not a.startswith("A"):
            if a.startswith("E"):  a = "Q17"
            if a.startswith("T"):  a = "Q17"
            new_list.append(a)

    i = 0
    for a in list:
        if (a.startswith("A")):
            final_list.append({a: temp_list[i]})
            # temp_dict[a] = temp_list[i]
            i += 1
    return final_list
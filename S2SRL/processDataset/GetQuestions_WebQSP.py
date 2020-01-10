# -*- coding: utf-8 -*-
import json
import os
import sys

from ..libbots import retriever_webqsp

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
            print("has union", id)

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
    return (action_item.e.startswith("m.") or action_item.e.startswith("?")) \
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


def calc_01_reward_type(target_value, gold_entities_set, type="jaccard"):
    true_reward = 0.0
    intersec = {}
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


# # for given questions, get similarity questions for each of them
# def get_RetrieverDict(path):
#     with open(path, "r", encoding='UTF-8') as dataset:
#         load_dict = json.load(dataset)
#         questions = load_dict["Questions"]
#         print(len(questions))
#         for q in questions:
#             question = q["ProcessedQuestion"]
#             Answers = []
#             id = q["QuestionId"]
#             answerList = q["Parses"][0]["Answers"]
#             for an in answerList:
#                 Answers.append(an['AnswerArgument'])
#             sparql = q["Parses"][0]["Sparql"]
#
#             retriever = Retriever_WebQSP(load_dict, {})
#             topNlist = retriever.Retrieve(5, question)


def get_right_answer_reorder_mask_file(path):
    # Load WebQuestions Semantic Parses
    WebQSPList = []
    WebQSPList_Correct = []
    WebQSPList_Incorrect = []
    json_errorlist = []
    true_count = 0

    errorlist = []
    with open(path, "r", encoding='UTF-8') as dataset:
        load_dictTrain = json.load(dataset)
        questions = load_dictTrain["Questions"]
        print(len(questions))

        # total rewards
        total_reward = 0
        test_count = 0
        total_reward_jaccard = 0
        total_reward_precision = 0
        total_reward_recall = 0

        for q in questions:
            question = q["ProcessedQuestion"]
            Answers = []
            id = q["QuestionId"]
            answerList = q["Parses"][0]["Answers"]
            for an in answerList:
                Answers.append(an['AnswerArgument'])
            sparql = q["Parses"][0]["Sparql"]
            mypair = Qapair(question, Answers, sparql)

            # if id == "WebQTest-1822": # test one
            if False:  # test all
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
                                for k, v in srt.items():
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
                                dict_entity = {e: "ENTITY{0}".format(e_index)}
                                entity_mask.update(dict_entity)
                                e_index += 1
                            for r in relation:
                                dict_relation = {r: "RELATION{0}".format(r_index)}
                                relation_mask.update(dict_relation)
                                r_index += 1
                            for t in type:
                                dict_type = {t: "TYPE{0}".format(t_index)}
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
                                for k, v in srt.items():
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
                                    mask_set = {a_mask: masklist}
                                    mask_action_sequence_list.append(mask_set)
                            if id != "" and question != "" and seq != "":
                                correct_item = WebQSP(id, question, seq, entity, relation, type, entity_mask,
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

        # jsondata = json.dumps(json_errorlist, indent=1)
        # fileObject = open('errorlist_full.json', 'w')
        # fileObject.write(jsondata)
        # fileObject.close()

        jsondata = json.dumps(WebQSPList_Correct, indent=1, default=WebQSP.obj_2_json)
        fileObject = open('right_answer_reorder_mask.json', 'w')
        fileObject.write(jsondata)
        fileObject.close()


if __name__ == "__main__":
    train_path = "WebQSP.train.json"
    test_path = "WebQSP.test.json"

    # get_RetrieverDict(train_path)

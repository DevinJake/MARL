# -*- coding: utf-8 -*-
# @Time    : 2019/4/8 21:02
# @Author  : Yaoleo
# @Blog    : yaoleo.github.io

# coding:utf-8
import json
from symbolics import Symbolics
from transform_util import transformBooleanToString, list2dict
import logging
log = logging.basicConfig(level = logging.INFO,
                           filename ='../../data/auto_QA_data/test_result/maml_newdata2k_reptile_retriever_joint_test_result.log',
                           filemode ='w', format = '%(message)s')

def transMask2ActionMAML(state, sample=False):
    predict_path = "../../data/saves/maml_newdata2k_reptile_retriever_joint/"
    if sample:
        predict_path += "sample_final_maml_predict.actions"
    else:
        predict_path += "final_maml_predict.actions"
    with open("../../data/auto_QA_data/CSQA_ANNOTATIONS_test.json", 'r') as load_f, \
            open(predict_path, 'r') as predict_actions:
        linelist = list()
        load_dict = json.load(load_f)
        num = 0
        total_precision = 0
        total_recall = 0
        total_right_count = 0
        total_answer_count = 0
        total_response_count = 0
        bool_right_count = 0
        count_right_count = 0
        for x in predict_actions:
            action = x.strip().split(":")[1]
            id = x.strip().split(":")[0]

            if id.startswith(state):
                num += 1
                entity_mask = load_dict[id]["entity_mask"] \
                    if load_dict[id]["entity_mask"] is not None else {}
                relation_mask = load_dict[id]["relation_mask"] \
                    if load_dict[id]["relation_mask"] is not None else {}
                type_mask = load_dict[id]["type_mask"] \
                    if load_dict[id]["type_mask"] is not None else {}
                response_entities = load_dict[id]["response_entities"].strip() \
                    if load_dict[id]["response_entities"] is not None else ""
                response_entities = response_entities.strip().split("|")
                orig_response = load_dict[id]["orig_response"].strip() \
                    if load_dict[id]["orig_response"] is not None else ""
                # Update(add) elements in dict.
                entity_mask.update(relation_mask)
                entity_mask.update(type_mask)
                new_action = list()
                # Default separator of split() method is any whitespace.
                for act in action.split():
                    for k, v in entity_mask.items():
                        if act == v:
                            act = k
                            break
                    new_action.append(act)
                print("{0}".format(num))
                '''print("{0}: {1}->{2}".format(num, id, action))'''
                logging.info("%d: %s -> %s", num, id, action)
                symbolic_seq = list2dict(new_action)
                symbolic_exe = Symbolics(symbolic_seq)
                answer = symbolic_exe.executor()

                if state.startswith("QuantitativeReasoning(Count)(All)") \
                        or state.startswith("ComparativeReasoning(Count)(All)"):
                    '''print (symbolic_seq)
                    print ("%s::%s" %(answer, orig_response))'''
                    logging.info(symbolic_seq)
                    logging.info("answer:%s, orig_response:%s", answer, orig_response)

                    if orig_response.isdigit() and answer == int(orig_response):
                        count_right_count += 1
                        '''print ("count_right_count+1")'''
                        logging.info("count_right_count+1")
                    else:
                        import re
                        orig_response = re.findall(r"\d+\.?\d*", orig_response)
                        orig_response = sum([int(i) for i in orig_response])
                        if answer == orig_response:
                            count_right_count += 1
                            '''print ("count_right_count+1")'''
                            logging.info("count_right_count+1")

                # For boolean, the returned answer is a list.
                if state.startswith("Verification(Boolean)(All)"):
                    # To judge the returned answers are in dict format or boolean format.
                    if type(answer) == dict:
                        temp = []
                        if '|BOOL_RESULT|' in answer:
                            temp.extend(answer['|BOOL_RESULT|'])
                            answer = temp
                            answer_string = transformBooleanToString(answer)
                            if answer_string!='' and answer_string == orig_response:
                                bool_right_count += 1
                                '''print("bool_right_count+1")'''
                                logging.info("bool_right_count+1")
                    else:
                        if answer:
                            answer = "YES"
                        if not answer:
                            answer = "NO"
                        if answer == orig_response:
                            bool_right_count += 1
                            '''print("bool_right_count+1")'''
                            logging.info("bool_right_count+1")

                # To judge the returned answers are in dict format or boolean format.
                if type(answer) == dict:
                    temp = []
                    if '|BOOL_RESULT|' in answer:
                        temp.extend(answer['|BOOL_RESULT|'])
                    else:
                        for key, value in answer.items():
                            if value:
                                temp.extend(list(value))
                    answer = temp

                elif type(answer) == type([]) or type(answer) == type(set([])):
                    answer = sorted((list(answer)))
                elif type(answer) == int:
                    answer = [answer]
                else:
                    answer = [answer]

                right_count = 0
                for e in response_entities:
                    if e in answer:
                        right_count += 1
                total_right_count += right_count
                total_answer_count += len(answer)
                total_response_count += len(response_entities)
                precision = right_count / float(len(answer)) if len(answer) != 0 else 0
                total_precision += precision
                recall = (right_count / float(len(response_entities))) if len(response_entities) != 0 else 0
                total_recall += recall
                '''print("orig:", len(response_entities), "answer:", len(answer), "right:", right_count)
                print("Precision:", precision),
                print("Recall:", recall)
                print('===============================')'''
                logging.info("orig:%d, answer:%d, right:%d", len(response_entities), len(answer), right_count)
                logging.info("Precision:%f", precision)
                logging.info("Recall:%f", recall)
                logging.info("============================")
            # print answer
        string_bool_right = "bool_right_count: %d" %bool_right_count
        string_count_right_count = "count_right_count: %d" %count_right_count
        string_total_num = "total_num::total_right::total_answer::total_response -> %d::%d::%d::%d" \
                           % (num, total_right_count, total_answer_count, total_response_count)
        print (string_bool_right)
        print (string_count_right_count)
        print (string_total_num)
        logging.info("bool_right_count:%d", bool_right_count)
        logging.info("count_right_count:%d", count_right_count)
        logging.info("total_num::total_right::total_answer::total_response -> %d::%d::%d::%d",
                     num, total_right_count, total_answer_count, total_response_count)
        linelist.append(string_bool_right + '\r\n')
        linelist.append(string_count_right_count + '\r\n')
        linelist.append(string_total_num + '\r\n')

        mean_pre = total_precision / num if num != 0 else 0.0
        mean_recall = total_recall / num if num != 0 else 0.0
        mean_pre2 = float(total_right_count) / total_answer_count if total_answer_count != 0 else 0.0
        mean_recall2 = float(total_right_count) / total_response_count if total_response_count != 0 else 0.0
        string_mean_pre = "state::mean_pre::mean_recall -> %s::%f::%f" %(state, mean_pre, mean_recall)
        string_mean_pre2 = "state::mean_pre2::mean_recall2 -> %s::%f::%f" %(state, mean_pre2, mean_recall2)
        print(string_mean_pre)
        print(string_mean_pre2)
        print("++++++++++++++")
        logging.info("state::mean_pre::mean_recall -> %s::%f::%f", state, mean_pre, mean_recall)
        logging.info("state::mean_pre2::mean_recall2 -> %s::%f::%f", state, mean_pre2, mean_recall2)
        logging.info("++++++++++++++")
        linelist.append(string_mean_pre + '\r\n')
        linelist.append(string_mean_pre2 + '\r\n')
        linelist.append('++++++++++++++\n\n')
        return linelist


def calculate_MAML_result(fila_path, sample=False):
    path = '../../data/auto_QA_data/test_result/'+fila_path+'.txt'
    linelist = list()
    fw = open(path, 'w', encoding="UTF-8")
    state_list = ["SimpleQuestion(Direct)", "Verification(Boolean)(All)", "QuantitativeReasoning(Count)(All)",
                  "QuantitativeReasoning(All)", "ComparativeReasoning(Count)(All)", "ComparativeReasoning(All)",
                  "LogicalReasoning(All)"]
    for state in state_list:
        linelist += transMask2ActionMAML(state, sample)
    fw.writelines(linelist)
    fw.close()


if __name__ == "__main__":
    # If testing the sample test dataset, set sample as True.
    # If testing the entire test dataset, set sample as False.
    calculate_MAML_result('maml_newdata2k_reptile_retriever_joint_test_result', sample=True)

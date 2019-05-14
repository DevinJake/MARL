# coding:utf-8
'''Get boolean logs from annotation file 'train_bool_all.txt'.
'''
from itertools import islice
import json

# To arrange the order of entities in input sequence.
def rearrangeEntities(context_entities, context):
    # TODO: A bug remain to be fixed.
    # If one entity is contained in another entity the alg is not correct.
    # For instance, 'Q910670|Q205576|Q33' in context 'Q333 does Q910670 share the border with Q205576'.
    entity_index = {x: (context.find(x) if context.find(x) != -1 else 100000) for x in context_entities}
    entity_index = sorted(entity_index.items(), key=lambda item:item[1])
    entities_output_string = ','.join([x[0].strip() for x in entity_index])
    # print (entity_index)
    # print (entities_output_string)
    return entities_output_string

# Get logs for boolean.
def getBooleanLogs():
    annotation_lines = list()
    fw = open('../../data/demoqa2/boolean.json', 'w', encoding="UTF-8")
    fwAuto = open('../../data/annotation_logs/bool_auto.log', 'w', encoding="UTF-8")
    fwOrig = open('../../data/annotation_logs/bool_orig.log', 'w', encoding="UTF-8")
    with open("../../data/demoqa2/train_bool_all.txt", 'r', encoding="UTF-8") as infile:
        count = 0
        while True:
            lines_gen = list(islice(infile, LINE_SIZE))
            if not lines_gen:
                break
            for line in lines_gen:
                annotation_lines.append(line.strip())
            count = count + 1
            print(count)
    line_index = 0
    question_dict = {}
    question_dict_dict = {}
    index = 0
    while line_index < len(annotation_lines):
        if annotation_lines[line_index].strip().isdigit():
            index = int(annotation_lines[line_index].strip())
        elif 'context_utterance:' in annotation_lines[line_index]:
            question_dict.setdefault('context_utterance', annotation_lines[line_index].split(':')[1].strip())
        elif 'context_relations:' in annotation_lines[line_index]:
            question_dict.setdefault('context_relations', annotation_lines[line_index].split(':')[1].strip())
        elif 'context_entities:' in annotation_lines[line_index]:
            question_dict.setdefault('context_entities', annotation_lines[line_index].split(':')[1].strip())
        elif 'context_types:' in annotation_lines[line_index]:
            question_dict.setdefault('context_types', annotation_lines[line_index].split(':')[1].strip())
        elif 'context:' in annotation_lines[line_index]:
            question_dict.setdefault('context', annotation_lines[line_index].split(':')[1].strip())
        elif 'orig_response:' in annotation_lines[line_index]:
            question_dict.setdefault('orig_response', annotation_lines[line_index].split(':')[1].strip())
        elif 'response_entities:' in annotation_lines[line_index]:
            question_dict.setdefault('response_entities', annotation_lines[line_index].split(':')[1].strip())
        elif 'CODE:' in annotation_lines[line_index]:
            question_dict.setdefault('CODE', list())
        elif 'symbolic_seq.append' in annotation_lines[line_index]:
            question_dict['CODE'].append(eval(annotation_lines[line_index].split('(')[1].strip().split(')')[0].strip()))
        elif '----------------------------' in annotation_lines[line_index]:
            if (line_index+1 < len(annotation_lines) and annotation_lines[line_index+1].strip().isdigit()) or (line_index+1 == len(annotation_lines)):
                question_dict_dict.setdefault(index, question_dict)
                question_dict = {}
                # print(line_index)
        line_index += 1
    fw.writelines(json.dumps(question_dict_dict, indent=1, ensure_ascii=False))
    fw.close()
    print("Get information from train_bool_all.txt!")

    question_index = 0
    fwOrig_lines = list()
    fwAuto_lines = list()
    for key, value in question_dict_dict.items():
        if len(value['CODE'])!=0:
            fwOrig_lines.append(str(question_index).strip()+'\n')
            fwOrig_lines.append('context_utterance:'+str(value['context_utterance']).strip()+'\n')
            fwAuto_lines.append(str(question_index).strip()+' '+str(value['context_utterance']).strip()+'\n')
            context_entities = [x.strip() for x in value['context_entities'].split('|')]
            context = str(value['context']).strip()
            fwOrig_lines.append('context_entities:'+rearrangeEntities(context_entities, context)+'\n')
            fwOrig_lines.append('context_relations:' + str(value['context_relations']).strip() + '\n')
            fwOrig_lines.append('context_types:' + str(value['context_types']).strip() + '\n')
            fwAuto_lines.append(str(value['CODE']) + '\n')
            question_index += 1
    fwAuto.writelines(fwAuto_lines)
    fwOrig.writelines(fwOrig_lines)
    fwAuto.close()
    fwOrig.close()
    print ("Writing logs is done!")

# Run to get the logs of manually annotation for boolean questions.
if __name__ == "__main__":
    # getTestDataset()
    getBooleanLogs()
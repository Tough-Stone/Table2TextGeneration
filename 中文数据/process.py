import json
import re


# 处理原始数据：删除表格中的空字段（已处理）
def delete_null(input_json):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i in range(len(data)):
            new_record = {}
            for k, v in data[i]['record'][0].items():
                if len(str(v)) > 0:
                    new_record[k] = str(v)
            del data[i]['record']
            data[i]['record'] = new_record
    return data


# 处理原始数据：表格中合并的项分开（已处理）
def split_table(input_json):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i in range(len(data)):
            for k, v in data[i]['record'].items():
                data[i]['record'][k] = re.split(r'[;、，,]', v)
    return data


# 替换中文逗号(已处理)
def replace_comma(input_json):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for d in data:
            d['text'] = d['text'].replace(',', '，')
            for k, v in d['record'].items():
                for i in range(len(v)):
                    v[i] = v[i].replace(',', '，')
    return data


# 获取表格（已获取）
def get_table(input_json):
    table = []
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for d in data:
            line = ''
            for k, v in d['record'].items():
                for i in range(len(v)):
                    line += '__'+k+'__' + ' : '
                    line += v[i] + ' '
                    line += '__EOS__' + ' '
            line = line.rstrip()
            table.append(line)
    return table


# 获取plan（已获取）
def get_plan(input_json):
    plan = []
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for j in range(len(data)):
            dict_list = []
            for k, v in data[j]['record'].items():
                for i in range(len(v)):
                    position = data[j]['text'].find(v[i])
                    if position >= 0:  # 如果存在
                        key_position = {'record_key': k, 'position': position}
                        dict_list.append(key_position)
            # 按出现位置排序
            dict_list_sort = sorted(dict_list, key=lambda x: x['position'], reverse=False)
            line = ''
            for dict in dict_list_sort:
                line += '__' + dict['record_key'] + '__ '
            line = line.rstrip()
            plan.append(line)
    return plan


# 获取参考句（已获取）
def get_sentence(input_json):
    sentence = []
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for d in data:
            sentence.append(d['text'])
    return sentence


# 获取special token的出现次数(已获取)
def get_special(input_json):
    special_token = {}
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for d in data:
            for k, v in d['record'].items():
                if k not in special_token:
                    special_token[k] = 0
                for i in range(len(v)):
                    special_token[k] += 1
    return special_token


if __name__ == '__main__':
    input_file = 'raw/dev.json'
    output_file = 'processed/special_token.txt'
    dev_dict = get_special('raw/dev.json')
    train_dict = get_special('raw/train.json')
    test_dict = get_special('raw/test.json')
    save_dict = {}
    for k, v in dev_dict.items():
        if k not in save_dict:
            save_dict[k] = v
        else:
            save_dict[k] += v
    for k, v in train_dict.items():
        if k not in save_dict:
            save_dict[k] = v
        else:
            save_dict[k] += v
    for k, v in test_dict.items():
        if k not in save_dict:
            save_dict[k] = v
        else:
            save_dict[k] += v

    sorted_dict = sorted(save_dict.items(), key=lambda x: x[1], reverse=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for k, v in sorted_dict:
            f.writelines('__'+k+'__ '+str(v)+'\n')

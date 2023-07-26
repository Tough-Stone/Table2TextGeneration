import json

import torch
from transformers import BartTokenizerFast, BartTokenizer, BartForConditionalGeneration
from generator.generator import Generator
from generator.dataclass import *
import time


class CPTTokenizer(BertTokenizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_mode = False

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        if not self.target_mode:
            return token_ids_0 + [self.eos_token_id]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def _switch_to_input_mode(self):
        self.target_mode = False

    def _switch_to_target_mode(self):
        self.target_mode = True

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.cls_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.sep_token_id


class JieBaTokenizer(CPTTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = partial(jieba.cut, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


def load_special_tokens(special_token_path, min_cnt):
    special_token_list, special_token_dict = [], {}
    with open(special_token_path, 'r', encoding='utf8') as i:
        lines = i.readlines()
        for l in lines:
            content_list = l.strip('\n').split()
            token = content_list[0]
            cnt = int(content_list[1])
            if cnt >= min_cnt:
                special_token_list.append(token)
                special_token_dict[token] = 1
    print(len(special_token_list))
    return special_token_list, special_token_dict


# 加载特殊token
special_token_path = r'中文数据/processed/special_token.txt'
min_cnt = 3
special_token_list, special_token_dict = load_special_tokens(special_token_path, min_cnt)

# 加载分词器
tokenizer = JieBaTokenizer.from_pretrained('中文预训练模型/t5-pegasus')
tokenizer.add_tokens(special_token_list)
decode_tokenizer = JieBaTokenizer.from_pretrained('中文预训练模型/t5-pegasus')
decode_tokenizer.add_tokens(special_token_list)

# 加载模型
model_ckpt = torch.load(r'generator/ckpt-chinese/pretrain/generator-pretrain.ckpt', map_location='cpu')
# model_ckpt = torch.load(r'ckpt/finetune/generator-rl-finetune.ckpt', map_location='cpu')
model_name = r'中文预训练模型/t5-pegasus'
tokenizer = decode_tokenizer
max_decode_len = 160
# 将模型参数读入Generator
model = Generator(model_name, tokenizer, max_decode_len, dropout=0.0)
model_parameters = model_ckpt['model']
model.load_state_dict(model_parameters)
model.eval()

# 加载数据
train_data_dict, dev_data_dict = {}, {}
train_data_dict['table_text_path'] = r'中文数据/processed/train/table.txt'
train_data_dict['content_text_path'] = r'中文数据/processed/train/plan.txt'
train_data_dict['reference_sentence_path'] = r'中文数据/processed/train/sentence.txt'
dev_data_dict['table_text_path'] = r'中文数据/processed/dev/table.txt'
dev_data_dict['content_text_path'] = r'中文数据/processed/dev/plan.txt'
dev_data_dict['reference_sentence_path'] = r'中文数据/processed/dev/sentence.txt'
max_table_len = 640
max_content_plan_len = 50
max_tgt_len = 160
min_slot_key_cnt = 10
start_time = time.time()
use_RL = False
data = Data(train_data_dict, dev_data_dict, max_table_len, max_content_plan_len, max_tgt_len,
            model_name, special_token_path, min_slot_key_cnt, use_RL)
print("--- %s seconds ---" % (time.time() - start_time))

table_path = r'中文数据/processed/test/table.txt'
content_path = r'中文数据/processed/test/plan.txt'
reference_path = r'中文数据/processed/test/sentence.txt'
# generated_plan_path = r'中文数据/processed/test/generated_plan.txt'

table_list, content_list, reference_list = [], [], []
with open(table_path, 'r', encoding='utf8') as i:
    lines = i.readlines()
    for l in lines:
        table_list.append(l.strip('\n'))

with open(content_path, 'r', encoding='utf8') as i:
    lines = i.readlines()
    for l in lines:
        content_list.append(l.strip('\n'))

with open(reference_path, 'r', encoding='utf8') as i:
    lines = i.readlines()
    for l in lines:
        reference_list.append(l.strip('\n'))

data_list = []
data_num = len(table_list)
for k in range(data_num):
    data_list.append((table_list[k], content_list[k], reference_list[k]))

# Generation with Reference Content Plan
save_result = []
# for idx in range(len(data_list)):
idx = 2
table, content_plan, reference_sentence = data_list[idx]
print('Table is:')
print(table + '\n')
print('Reference Content Plan is:')
print(content_plan + '\n')
print('Reference Sentence is:')
print(reference_sentence + '\n')

src_id_list = [data.sep_idx] + data.load_one_text_id(table, 640) + [data.sep_idx] + \
              data.load_one_text_id(content_plan, 50)

batch_src_id_list = [src_id_list]

batch_src_tensor, batch_src_mask = data.process_source_tensor(batch_src_id_list)

result = model.generate(batch_src_tensor, batch_src_mask)[0]
print('Generated Result is:')
print(result + '\n')
new_json = {'Reference': reference_sentence, 'Generated': result}
save_result.append(new_json)

# with open('plangen.json', 'w', encoding='utf-8') as f:
#     json.dump(save_result, f, ensure_ascii=False, indent=4)

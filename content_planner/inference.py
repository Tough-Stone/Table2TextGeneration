import sys

from content_planner.utlis import load_special_tokens
from content_planner.contentplanner import ContentPlanner
import torch

# 加载special token
path = r'../中文数据/processed/special_token.txt'
min_slot_cnt = 10
special_token_list = load_special_tokens(path, min_slot_cnt)
# 加载模型
model_name, special_token_list = '../中文预训练模型/bert-base-chinese', special_token_list
model = ContentPlanner(model_name, special_token_list=special_token_list)
ckpt_path = r'./ckpt-chinese'  # the path specified in the --save_path_prefix argument of the training script
model.load_pretrained_model(ckpt_path)
model.eval()

# 读取测试集table信息
with open('../中文数据/processed/test/table.txt', 'r', encoding='utf-8') as f:
    tables = f.readlines()

# 获取预测plan
genrated_plan = []
for table in tables:
    # prepare table id list
    table_id_list = model.tokenizer.encode(table, max_length=320, truncation=True, add_special_tokens=False)[:320]
    cls_token_id, sep_token_id = model.tokenizer.cls_token_id, model.tokenizer.sep_token_id
    table_id_list = [cls_token_id] + table_id_list + [sep_token_id]

    src_tensor = torch.LongTensor(table_id_list).view(1, -1)
    # prepare selected content plan id list
    selected_id_list = [model.targettokenizer.extract_selective_ids(table.strip('\n').strip())]
    # candidate_set = model.targettokenizer.convert_ids_to_text(selected_id_list[0])
    # print('The candidate set is: {}\n'.format(candidate_set))

    # make prediction
    predicted_content_plan = model.selective_decoding(src_tensor, selected_id_list)
    print('The pedicted content plan is: {}'.format(predicted_content_plan))
    genrated_plan.append(predicted_content_plan)


with open('../中文数据/processed/test/generated_plan.txt', 'w', encoding='utf-8') as f:
    for plan in genrated_plan:
        plan = str(plan)[2:-2]
        f.write(plan+'\n')

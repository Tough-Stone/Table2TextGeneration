import nltk

nltk.download('stopwords')
nltk.download('punkt')
import os
import torch
from operator import itemgetter
from transformers import AdamW, get_linear_schedule_with_warmup
import argparse
from dataclass import Data
from generator import Generator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--train_table_text_path', type=str)
    parser.add_argument('--train_content_text_path', type=str)
    parser.add_argument('--train_reference_sentence_path', type=str)
    parser.add_argument('--dev_table_text_path', type=str)
    parser.add_argument('--dev_content_text_path', type=str)
    parser.add_argument('--dev_reference_sentence_path', type=str)
    parser.add_argument('--special_token_path', type=str)
    parser.add_argument('--max_table_len', type=int, default=640)
    parser.add_argument('--max_content_plan_len', type=int, default=50)
    parser.add_argument('--max_tgt_len', type=int, default=160)
    parser.add_argument('--min_slot_key_cnt', type=int, default=3)
    # model configuration
    parser.add_argument('--model_name', type=str, default='../中文预训练模型/t5-pegasus')
    parser.add_argument('--max_decode_len', type=int, default=160)
    # learning configuration
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--total_steps', type=int, default=200000)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--print_every', type=int, default=200)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--ckpt_path', type=str, default=r'./ckpt-chinese/pretrain/')
    return parser.parse_args()


def map_cuda(tensor_item, device, is_cuda):
    res_list = []
    if is_cuda:
        res_list.append(tensor_item[0].cuda(device))
        res_list.append(tensor_item[1].cuda(device))
    else:
        res_list = tensor_item
    return res_list


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('Cuda is available.')
    cuda_available = torch.cuda.is_available()

    args = parse_config()
    device = args.gpu_id

    test_output_dir = args.ckpt_path
    import os

    if os.path.exists(test_output_dir):
        pass
    else:  # recursively construct directory
        os.makedirs(test_output_dir, exist_ok=True)

    print('Start loading data...')
    train_dict, dev_dict = {}, {}
    train_dict['table_text_path'] = args.train_table_text_path
    train_dict['reference_sentence_path'] = args.train_reference_sentence_path
    dev_dict['table_text_path'] = args.dev_table_text_path
    dev_dict['reference_sentence_path'] = args.dev_reference_sentence_path
    train_dict['content_text_path'] = args.train_content_text_path
    dev_dict['content_text_path'] = args.dev_content_text_path
    special_token_name = args.special_token_path

    train_dict['processed_file_path'] = None
    dev_dict['processed_file_path'] = None
    use_RL = False

    data = Data(train_dict, dev_dict, args.max_table_len, args.max_content_plan_len, args.max_tgt_len,
                args.model_name, special_token_name, args.min_slot_key_cnt, use_RL)
    print('Data loaded.')

    model = Generator(model_name=args.model_name, tokenizer=data.decode_tokenizer,
                      max_decode_len=args.max_decode_len, dropout=args.dropout)

    if torch.cuda.is_available():
        model = model.cuda(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_update_steps = (args.total_steps // args.gradient_accumulation_steps) + 1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.total_steps)
    optimizer.zero_grad()

    train_num, dev_num = data.train_num, data.dev_num
    batch_size = args.batch_size
    train_step_num, dev_step_num = int(train_num / batch_size) + 1, int(dev_num / batch_size) + 1

    batches_processed = 0
    max_dev_score = 0.
    total_steps = args.total_steps
    print_every, eval_every = args.print_every, args.eval_every

    train_loss_accumulated = 0.
    min_loss = 100

    log_path = args.ckpt_path + '/log.txt'
    model.train()
    for one_step in range(total_steps):
        epoch = one_step // train_step_num
        batches_processed += 1

        _, _, train_batch_src_item, train_batch_tgt_item, _ = data.get_next_train_batch(batch_size)

        train_batch_src_tensor, train_batch_src_mask = map_cuda(train_batch_src_item, device, cuda_available)
        train_batch_tgt_in_tensor, train_batch_tgt_out_tensor = map_cuda(train_batch_tgt_item, device, cuda_available)

        train_loss = model(train_batch_src_tensor, train_batch_src_mask, train_batch_tgt_in_tensor,
                           train_batch_tgt_out_tensor)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        train_loss_accumulated += train_loss.item()

        if (one_step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if batches_processed % print_every == 0:
            curr_train_loss = train_loss_accumulated / print_every
            train_log_text = 'At epoch {}, batch {}, train loss {}, max combine score is {}'.format(
                epoch, batches_processed, curr_train_loss, max_dev_score)
            print(train_log_text)
            with open(log_path, 'a', encoding='utf8') as logger:
                logger.writelines(train_log_text + '\n')
            train_loss_accumulated = 0.

        if batches_processed % eval_every == 0:
            model.eval()
            dev_loss_accumulated = 0.
            dev_output_text_list = []
            print('Start evaluation...')
            with torch.no_grad():
                import progressbar

                p = progressbar.ProgressBar(dev_step_num)
                p.start()
                for dev_step in range(dev_step_num):
                    p.update(dev_step)
                    _, _, dev_batch_src_item, dev_batch_tgt_item, _ = data.get_next_dev_batch(batch_size)
                    dev_batch_src_tensor, dev_batch_src_mask = map_cuda(dev_batch_src_item, device, cuda_available)
                    dev_batch_tgt_in_tensor, dev_batch_tgt_out_tensor = map_cuda(dev_batch_tgt_item, device,
                                                                                 cuda_available)
                    dev_loss = model(dev_batch_src_tensor, dev_batch_src_mask, dev_batch_tgt_in_tensor,
                                     dev_batch_tgt_out_tensor)
                    dev_loss_accumulated += dev_loss.item()

                    decoded_result = model.generate(dev_batch_src_tensor, dev_batch_src_mask)
                    dev_output_text_list += decoded_result
                p.finish()

                dev_output_text_list = dev_output_text_list[:dev_num]
                dev_text_out_path = './test_out.txt'
                with open(dev_text_out_path, 'w', encoding='utf8') as o:
                    for text in dev_output_text_list:
                        o.writelines(text + '\n')

                # overall_bleu, overlap_bleu, nonoverlap_bleu = eval_totto(dev_text_out_path, args.dev_reference_path)

                # one_dev_combine_score = overall_bleu + overlap_bleu + nonoverlap_bleu
                one_dev_loss = dev_loss_accumulated / dev_step_num
                print('----------------------------------------------------------------')
                valid_log_text = 'At epoch {}, batch {}, dev loss is {}'.format(epoch, batches_processed, one_dev_loss)
                print(valid_log_text)
                with open(log_path, 'a', encoding='utf8') as logger:
                    logger.writelines(valid_log_text + '\n')
                save_name = '/generator-pretrain.ckpt'
                print('----------------------------------------------------------------')

                if one_dev_loss < min_loss:
                    torch.save({'model': model.state_dict()}, test_output_dir + save_name)
                    print('save, one_dev_loss = {}'.format(one_dev_loss))
                    with open(log_path, 'a', encoding='utf8') as logger:
                        logger.writelines('save, one_dev_loss = {}'.format(one_dev_loss) + '\n')
                    min_loss = one_dev_loss
                else:
                    pass

                fileData = {}
                for fname in os.listdir(test_output_dir):
                    if fname.startswith('generator'):
                        fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                    else:
                        pass
                sortedFiles = sorted(fileData.items(), key=itemgetter(1))
                if len(sortedFiles) < 1:
                    pass
                else:
                    delete = len(sortedFiles) - 1
                    for x in range(0, delete):
                        os.remove(test_output_dir + '/' + sortedFiles[x][0])
            model.train()

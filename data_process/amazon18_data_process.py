# import argparse
# import collections
# import gzip
# import html
# import json
# import os
# import random
# import re
# import torch
# from tqdm import tqdm
# import numpy as np
# from utils import check_path, clean_text, amazon18_dataset2fullname, write_json_file, write_remap_index, load_json

# def load_ratings(file, images_info):
#     users, items, inters = set(), set(), set()
#     with open(file, 'r') as fp:
#         for line in tqdm(fp, desc='Load ratings'):
#             try:
#                 item, user, rating, time = line.strip().split(',')
#                 if item in images_info and len(images_info[item]) != 0:
#                     users.add(user)
#                     items.add(item)
#                     inters.add((user, item, float(rating), int(time)))
#             except ValueError:
#                 print(line)
#     return users, items, inters


# def load_meta_items(file):
#     items = {}
#     with gzip.open(file, "r") as fp:
#         for line in tqdm(fp, desc="Load metas"):
#             data = json.loads(line)
#             item = data["asin"]
#             title = clean_text(data["title"])

#             descriptions = data["description"]
#             descriptions = clean_text(descriptions)

#             brand = data["brand"].replace("by\n", "").strip()

#             categories = data["category"]
#             new_categories = []
#             for category in categories:
#                 if "</span>" in category:
#                     break
#                 new_categories.append(category.strip())
#             categories = ",".join(new_categories).strip()

#             items[item] = {"title": title, "description": descriptions, "brand": brand, "categories": categories}
#             # print(items[item])
#     return items


# def load_review_data(args, user2id, item2id):

#     dataset_full_name = amazon18_dataset2fullname[args.dataset]
#     review_file_path = os.path.join(args.input_path, 'Review', dataset_full_name + '.json.gz')

#     reviews = {}

#     with gzip.open(review_file_path, "r") as fp:

#         for line in tqdm(fp,desc='Load reviews'):
#             inter = json.loads(line)
#             try:
#                 user = inter['reviewerID']
#                 item = inter['asin']
#                 if user in user2id and item in item2id:
#                     uid = user2id[user]
#                     iid = item2id[item]
#                 else:
#                     continue
#                 if 'reviewText' in inter:
#                     review = clean_text(inter['reviewText'])
#                 else:
#                     review = ''
#                 if 'summary' in inter:
#                     summary = clean_text(inter['summary'])
#                 else:
#                     summary = ''
#                 reviews[str((uid,iid))]={"review":review, "summary":summary}

#             except ValueError:
#                 print(line)

#     return reviews


# def get_user2count(inters):
#     user2count = collections.defaultdict(int)
#     for unit in inters:
#         user2count[unit[0]] += 1
#     return user2count


# def get_item2count(inters):
#     item2count = collections.defaultdict(int)
#     for unit in inters:
#         item2count[unit[1]] += 1
#     return item2count


# def generate_candidates(unit2count, threshold):
#     cans = set()
#     for unit, count in unit2count.items():
#         if count >= threshold:
#             cans.add(unit)
#     return cans, len(unit2count) - len(cans)


# def filter_inters(inters, can_items=None,
#                   user_k_core_threshold=0, item_k_core_threshold=0):
#     new_inters = []

#     # filter by meta items
#     if can_items:
#         print('\nFiltering by meta items: ')
#         for unit in tqdm(inters):
#             if unit[1] in can_items.keys():
#                 new_inters.append(unit)
#         inters, new_inters = new_inters, []
#         print('    The number of inters: ', len(inters))

#     # filter by k-core
#     if user_k_core_threshold or item_k_core_threshold:
#         print('\nFiltering by k-core:')
#         idx = 0
#         user2count = get_user2count(inters)
#         item2count = get_item2count(inters)

#         while True:
#             new_user2count = collections.defaultdict(int)
#             new_item2count = collections.defaultdict(int)
#             users, n_filtered_users = generate_candidates( # users is set
#                 user2count, user_k_core_threshold)
#             items, n_filtered_items = generate_candidates(
#                 item2count, item_k_core_threshold)
#             if n_filtered_users == 0 and n_filtered_items == 0:
#                 break
#             for unit in inters:
#                 if unit[0] in users and unit[1] in items:
#                     new_inters.append(unit)
#                     new_user2count[unit[0]] += 1
#                     new_item2count[unit[1]] += 1
#             idx += 1
#             inters, new_inters = new_inters, []
#             user2count, item2count = new_user2count, new_item2count
#             print('    Epoch %d The number of inters: %d, users: %d, items: %d'
#                     % (idx, len(inters), len(user2count), len(item2count)))
#     return inters


# def make_inters_in_order(inters):
#     user2inters, new_inters = collections.defaultdict(list), list()
#     for inter in tqdm(inters):
#         user, item, rating, timestamp = inter
#         user2inters[user].append((user, item, rating, timestamp))
#     for user in tqdm(user2inters):
#         user_inters = user2inters[user]
#         user_inters.sort(key=lambda d: d[3])
#         interacted_item = set()
#         for inter in user_inters:
#             if inter[1] in interacted_item: # 过滤重复交互
#                 continue
#             interacted_item.add(inter[1])
#             new_inters.append(inter)
#     return new_inters


# def preprocess_rating(args):
#     dataset_full_name = amazon18_dataset2fullname[args.dataset]

#     print('Process rating data: ')
#     print(' Dataset: ', args.dataset)
    
#     images_info_file = os.path.join(args.input_path, 'Images', f'{args.dataset}_images_info.json')
#     images_info = load_json(images_info_file)


#     # load ratings
#     rating_file_path = os.path.join(args.input_path, 'Ratings', dataset_full_name + '.csv')
#     rating_users, rating_items, rating_inters = load_ratings(rating_file_path, images_info)

#     # load item IDs with meta data
#     meta_file_path = os.path.join(args.input_path, 'Metadata', f'meta_{dataset_full_name}.json.gz')
#     meta_items = load_meta_items(meta_file_path)

#     # 1. Filter items w/o meta data;
#     # 2. K-core filtering;
#     print('The number of raw inters: ', len(rating_inters))

#     rating_inters = make_inters_in_order(rating_inters)

#     rating_inters = filter_inters(rating_inters, can_items=meta_items,
#                                   user_k_core_threshold=args.user_k,
#                                   item_k_core_threshold=args.item_k)

#     # sort interactions chronologically for each user
#     rating_inters = make_inters_in_order(rating_inters)
#     print('\n')

#     # return: list of (user_ID, item_ID, rating, timestamp)
#     return rating_inters, meta_items

# def convert_inters2dict(inters):
#     user2items = collections.defaultdict(list)
#     user2index, item2index = dict(), dict()
#     for inter in inters:
#         user, item, rating, timestamp = inter
#         if user not in user2index:
#             user2index[user] = len(user2index)
#         if item not in item2index:
#             item2index[item] = len(item2index)
#         user2items[user2index[user]].append(item2index[item])
#     return user2items, user2index, item2index

# def generate_data(args, rating_inters):
#     print('Split dataset: ')
#     print(' Dataset: ', args.dataset)

#     # generate train valid temp
#     user2items, user2index, item2index = convert_inters2dict(rating_inters)
#     train_inters, valid_inters, test_inters = dict(), dict(), dict()
#     for u_index in range(len(user2index)):
#         inters = user2items[u_index]
#         # leave one out
#         train_inters[u_index] = [str(i_index) for i_index in inters[:-2]]
#         valid_inters[u_index] = [str(inters[-2])]
#         test_inters[u_index] = [str(inters[-1])]
#         assert len(user2items[u_index]) == len(train_inters[u_index]) + \
#                len(valid_inters[u_index]) + len(test_inters[u_index])
#     return user2items, train_inters, valid_inters, test_inters, user2index, item2index

# def convert_to_atomic_files(args, train_data, valid_data, test_data):
#     print('Convert dataset: ')
#     print(' Dataset: ', args.dataset)
#     uid_list = list(train_data.keys())
#     uid_list.sort(key=lambda t: int(t))

#     with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.train.inter'), 'w') as file:
#         file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
#         for uid in uid_list:
#             item_seq = train_data[uid]
#             seq_len = len(item_seq)
#             for target_idx in range(1, seq_len):
#                 target_item = item_seq[-target_idx]
#                 seq = item_seq[:-target_idx][-50:]
#                 file.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')

#     with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.valid.inter'), 'w') as file:
#         file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
#         for uid in uid_list:
#             item_seq = train_data[uid][-50:]
#             target_item = valid_data[uid][0]
#             file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

#     with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.test.inter'), 'w') as file:
#         file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
#         for uid in uid_list:
#             item_seq = (train_data[uid] + valid_data[uid])[-50:]
#             target_item = test_data[uid][0]
#             file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='Instruments', help='Instruments / Arts / Games')
#     parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
#     parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
#     parser.add_argument('--input_path', type=str, default='/datasets/datasets/amazon18')
#     parser.add_argument('--output_path', type=str, default='/datasets/datasets/LC-Rec_image')
#     return parser.parse_args()


# if __name__ == '__main__':
#     args = parse_args()

#     from utils import amazon18_dataset_list
    
#     # amazon18_dataset_list = ['Cell', 'Food', 'Movies', 'Pet']
#     # amazon18_dataset_list = ['Scientific', 'Pantry', 'Office']
#     # amazon18_dataset_list = ['Automotive', 'CDs', 'Electronics', 'Sports', 'Tools', 'Toys']
#     amazon18_dataset_list = ['Instruments']
#     for dataset in amazon18_dataset_list:
        
#         print('\n' + '=' * 20 + '\n')
#         if dataset == 'Fashion':
#             continue
#         args.dataset = dataset
        
#         # load interactions from raw rating file
#         rating_inters, meta_items = preprocess_rating(args)

        
#         # split train/valid/temp
#         all_inters, train_inters, valid_inters, test_inters, user2index, item2index = generate_data(args, rating_inters)

#         check_path(os.path.join(args.output_path, args.dataset))

#         write_json_file(all_inters, os.path.join(args.output_path, args.dataset, f'{args.dataset}.inter.json'))
#         convert_to_atomic_files(args, train_inters, valid_inters, test_inters)

#         item2feature = collections.defaultdict(dict)
#         for item, item_id in item2index.items():
#             item2feature[item_id] = meta_items[item]

#         # reviews = load_review_data(args, user2index, item2index)

#         print("user:",len(user2index))
#         print("item:",len(item2index))

#         write_json_file(item2feature, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item.json'))
#         # write_json_file(reviews, os.path.join(args.output_path, args.dataset, f'{args.dataset}.review.json'))


#         write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2id'))
#         write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2id'))



import argparse
import torch
from tqdm import tqdm
import numpy as np
import os
from utils import *
from transformers import AutoTokenizer, AutoModel

def generate_item_embedding(args, item_text_list, tokenizer, model, word_drop_ratio=-1):
    print(f'Generate Text Embedding: ')
    print(f' Dataset: {args.dataset}')
    print(f' Device: {args.device}')
    print(f' Model is on: {next(model.parameters()).device}')
    
    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item] = text
    for text in order_texts:
        assert text != [0]

    embeddings = []
    batch_size = 16  # 从小的batch size开始
    model.eval()
    
    with torch.no_grad():
        for start in tqdm(range(0, len(order_texts), batch_size), desc="Generating embeddings"):
            field_texts = order_texts[start: start + batch_size]
            field_texts = list(zip(*field_texts))
    
            field_embeddings = []
            for sentences in field_texts:
                sentences = list(sentences)
                
                # 编码并移动到GPU
                encoded_sentences = tokenizer(
                    sentences, 
                    max_length=args.max_sent_len,
                    truncation=True, 
                    return_tensors='pt',
                    padding=True
                ).to(args.device)  # 确保在GPU上
                
                # 模型推理
                outputs = model(
                    input_ids=encoded_sentences.input_ids,
                    attention_mask=encoded_sentences.attention_mask
                )
    
                # 计算平均
                masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
                mean_output = masked_output.sum(dim=1) / encoded_sentences['attention_mask'].sum(dim=-1, keepdim=True)
                mean_output = mean_output.detach().cpu()
                field_embeddings.append(mean_output)
    
            field_mean_embedding = torch.stack(field_embeddings, dim=0).mean(dim=0)
            embeddings.append(field_mean_embedding)

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    file = os.path.join(args.root, args.dataset + '.emb-' + args.plm_name + "-td" + ".npy")
    np.save(file, embeddings)
    print(f'Saved to: {file}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments')
    parser.add_argument('--root', type=str, default="data_process/MQL4GRec")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--plm_name', type=str, default='llama')
    parser.add_argument('--model_name_or_path', type=str, default='huggyllama/llama-7b')
    parser.add_argument('--model_cache_dir', type=str, default='/scratch/zl4789/MQL4GRec/.cachemodels')
    parser.add_argument('--max_sent_len', type=int, default=512)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # ============ 诊断 GPU ============
    print("="*60)
    print("GPU Diagnostics:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("  WARNING: CUDA is NOT available!")
    print(f"  PyTorch version: {torch.__version__}")
    print("="*60)
    
    # ============ 设置设备 ============
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("\n✗ Using CPU (GPU not available)")
        
    args.device = device
    args.root = os.path.join(args.root, args.dataset)
    
    # ============ 加载数据 ============
    print("\n" + "="*60)
    print("Loading data...")
    from amazon_text_emb import preprocess_text
    item_text_list = preprocess_text(args)
    print(f"Loaded {len(item_text_list)} items")
    
    # ============ 加载模型 ============
    print("\n" + "="*60)
    print(f"Loading model: {args.model_name_or_path}")
    kwargs = {"cache_dir": args.model_cache_dir}
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    
    print("Loading model weights...")
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,  # 使用FP16
        device_map=None,  # 不使用自动设备映射
        low_cpu_mem_usage=True,
        **kwargs
    )
    
    # ============ 确保模型在GPU上 ============
    print(f"Moving model to {device}...")
    model = model.to(device)
    model.eval()
    
    print(f"Model is now on: {next(model.parameters()).device}")
    print("="*60)
    
    # 测试GPU是否工作
    if torch.cuda.is_available():
        print("\n🔍 Testing GPU computation...")
        test_tensor = torch.randn(10, 10).to(device)
        test_output = test_tensor @ test_tensor
        print("✓ GPU computation test passed!")
        print(f"  Current GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # ============ 生成 embeddings ============
    print("\n" + "="*60)
    generate_item_embedding(args, item_text_list, tokenizer, model)
    
    if torch.cuda.is_available():
        print(f"\nMax GPU memory used: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    
    print("\n✓ Done!")
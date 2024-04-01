import os
import json
import random
import argparse

import tqdm

import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GPTNeoForCausalLM
from datasets import load_dataset, DatasetDict, Dataset

from countdown_utils import *
from countdown_bfs import bfs
from countdown_dfs import dfs

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--ckpt", type=str, help="path to checkpoint")
parser.add_argument("-n", "--num",type=int, default=10)
parser.add_argument("-o", "--offset",type=int, default=0)
parser.add_argument("--data_dir", type=str, default="data/")
parser.add_argument("-d", "--data",type=str, default="val_b3_t100_n100000_random.json")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--ctx", type=int, default=4096)
parser.add_argument("--gens", type=int, default=1)


def eval_ll(model, tokenizer, data, batch_size=128, context_len=4096, temperature=0.0, n=1):
    """
    Evaluate the model on the data using a sliding window so that the context length is not exceeded
    """
    output_texts_concat = []
    for b in tqdm.trange(0, len(data), batch_size):
        batch = data[b:min(b+batch_size, len(data))]
        output_texts = ["" for _ in range(len(batch))]
        tokenizer.padding_side = "left"
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        inputs = inputs['input_ids']
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):

        if n == 1:
            if temperature == 0.0:
                outputs = model.generate(input_ids=inputs, pad_token_id=tokenizer.eos_token_id, attention_mask=torch.ones_like(inputs), max_length=context_len, num_beams=1, do_sample=False)
            else:
                outputs = model.generate(input_ids=inputs, pad_token_id=tokenizer.eos_token_id, attention_mask=torch.ones_like(inputs), max_length=context_len, num_beams=1, do_sample=True, temperature=temperature)
            # split output vector into first N tokens and the rest
            output_tokens = outputs
            output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
            tokenizer.padding_side = "left"
            output_texts = [ot + ot_now for ot, ot_now in zip(output_texts, output_text)]
            # print token lens of tokenized outputs
            print([len(tokenizer(ot)['input_ids']) for ot in output_texts])
            output_texts_concat += output_texts
        else:
            assert temperature > 0.0, "Temperature must be greater than 0 for sampling"
            all_outputs = []
            all_ratings = []
            for i in range(n):
                outputs = model.generate(input_ids=inputs, pad_token_id=tokenizer.eos_token_id, attention_mask=torch.ones_like(inputs), max_length=context_len, do_sample=True, temperature=temperature)
                output_tokens = outputs
                output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
                tokenizer.padding_side = "left"
                # get rating for each output
                ratings = [metric_fn(ot.split(tokenizer.bos_token)[1], mode="sft")[0] for ot in output_text]
                all_ratings.append(ratings)
                all_outputs.append(output_text)
            # only keep the output with the highest rating for each input
            all_ratings = np.array(all_ratings)

            print(all_ratings)
            print(f"average rating", np.mean(all_ratings))
            # all ratings is n x batch_size
            max_ratings = np.argmax(all_ratings, axis=0)
            max_rating_vals = np.max(all_ratings, axis=0)
            print(f"max ratings", np.mean(max_rating_vals))
            # max ratings is batch_size, output_texts is n x batch_size
            output_texts = [all_outputs[max_r][i] for i, max_r in enumerate(max_ratings)]
            output_texts_concat += output_texts
    return output_texts_concat 

args = parser.parse_args()
torch.manual_seed(args.seed)
model = GPTNeoForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')

model.eval()
model.cuda()

tokenizer = AutoTokenizer.from_pretrained(args.ckpt, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

data_file = os.path.join(args.data_dir, args.data)

with open(data_file, "r") as json_file:
    data = json.load(json_file)

predictions = []
pred_ratings = []
pred_reasons = []
tokenizer.padding_side = "left"
test_prompts = [tokenizer.bos_token + f"Current State: {sample['target']}:{sample['nums']}, Operations: []"  for sample in data[args.offset:args.num]]
len_nums = [len(sample['nums']) for sample in data[args.offset:args.num]]
data_4 = [d for d, l in zip(test_prompts, len_nums) if l == 4]
predictions = eval_ll(model, tokenizer, data_4, batch_size=args.batch_size, context_len=args.ctx, temperature=args.temperature, n=args.gens)

len_pred_nums = [4 for _ in predictions]

# rate outputs
true_rating = []
for i in range(len(predictions)):
    rating, reason = metric_fn(predictions[i].split(tokenizer.bos_token)[1], mode="sft")
    tr, _ = metric_fn(f"{data[i]['search_path']}", mode="sft")
    pred_ratings.append(rating)
    true_rating.append(tr)
    pred_reasons.append(reason)

# get max rating for each sample with its index
pred_ratings = np.array(pred_ratings)

# print results
print("Results Summary:")
print(f"Average rating: {np.mean(pred_ratings)}")
print(f"Average true rating: {np.mean(true_rating)}")
print(f"Accuracy: {np.mean([r > 0 for r in pred_ratings])}")
print(f"True Accuracy: {np.mean([r > 0 for r in true_rating])}")

ckpt_dir = os.path.dirname(args.ckpt)
# save results
results_file = os.path.join(ckpt_dir, f"results_{args.data.replace('/','_')}_{args.num}_{args.offset}")
with open(results_file, "w") as f:
    json.dump({"trajectories": predictions, "ratings": pred_ratings.tolist(), "reasons": pred_reasons}, f, indent=4)

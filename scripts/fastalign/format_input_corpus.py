"""
Author: Guanlin Li
Date  : Jan. 7 2019
"""

import argparse
import os
import os.path as path

parser = argparse.ArgumentParser("Convert fast_align format corpus from two aligned corpora")

parser.add_argument('--src-corpus', type=str, required=True)
parser.add_argument('--tgt-corpus', type=str, required=True)
parser.add_argument('--save-to', type=str, required=True)

args = parser.parse_args()
parent_dir = '/'.join(os.path.abspath(args.src_corpus).split('/')[:-1])
fa_dir = path.join(parent_dir, 'fastalign')
if not path.exists(fa_dir):
    os.mkdir(fa_dir)
# saveto = path.join(fa_dir, args.save_to)
saveto = args.save_to
print('Merging two corpora into one...')
with open(args.src_corpus, 'r', encoding='utf-8') as f_src:
    with open(args.tgt_corpus, 'r', encoding='utf-8') as f_tgt:
        with open(saveto, 'w', encoding='utf-8') as f_saveto:
            for (src_line, tgt_line) in zip(f_src, f_tgt):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                new_line = src_line + ' ||| ' + tgt_line + '\n'
                f_saveto.write(new_line)
print('Done.')

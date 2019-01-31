"""
Author: Guanlin Li
Date  : Jan. 31 2019
"""

import argparse
import random

parser = argparse.ArgumentParser('Build augmented only corpus 20x '
                                'larger than original corpus')
parser.add_argument('--align', type=str, required=True,
                    help='alignment files')
parser.add_argument('--src', type=str, required=True,
                    help='source file')
parser.add_argument('--tgt', type=str, required=True,
                    help='target file')
parser.add_argument('--repeat', type=int, required=True,
                    help='number of times bigger than the origin')

args = parser.parse_args()

alignments = []
with open(args.align, 'r', encoding='utf-8') as f_align:
    for line in f_align:
        aligns = line.strip().split()  # m-n:p-q
        alignments.append(aligns)

src_corpus = []
tgt_corpus = []

with open(args.src, 'r', encoding='utf-8') as f_src, open(args.tgt, 'r', encoding='utf-8') as f_tgt:
    for s, t in zip(f_src, f_tgt):
        s = s.strip()
        t = t.strip()
        src_corpus.append(s)
        tgt_corpus.append(t)

# assert equal size
assert len(src_corpus) == len(tgt_corpus) == len(alignments),\
        'Size of src, tgt, alignments should be equal'

saveto_src_path = args.src + '.aug'
saveto_tgt_path = args.tgt + '.aug'
saveto_src_corpus = []  # saveto_corpus should be 20x more than origin
saveto_tgt_corpus = []
for i in range(args.repeat):
    for s, t, a in zip(src_corpus, tgt_corpus, alignments):
        s = s.strip().split()  # a list of tokens
        t = t.strip().split()

        # get aligned phrase pair, through uniform sampling
        num_align = len(a)
        sampled_a = a[random.randint(0, num_align - 1)]
        m_n, p_q = sampled_a.split(':')
        m, n = m_n.split('-')
        p, q = p_q.split('-')
        m, n = int(m), int(n)
        p, q = int(p), int(q)
        proto_s = [s_i for s_i in s]
        proto_t = [t_i for t_i in t]
        for i in range(m, n + 1):
            proto_s[i] = '<xxx>'
        for j in range(p, q + 1):
            proto_t[j] = '<xxx>'

        # add to saveto src/tgt corpus
        saveto_src_corpus.append(proto_s)
        saveto_tgt_corpus.append(proto_t)

with open(saveto_src_path, 'w', encoding='utf-8') as f_proto_src:
    with open(saveto_tgt_path, 'w', encoding='utf-8') as f_proto_tgt:
        for s, t in zip(saveto_src_corpus, saveto_tgt_corpus):
            f_proto_src.write(' '.join(s) + '\n')
            f_proto_tgt.write(' '.join(t) + '\n')

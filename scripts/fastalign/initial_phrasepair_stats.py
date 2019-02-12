"""
Author: Guanlin Li
Date  : Feb. 12 2019
Usage :
    Get the avg phrase pair count for different lengths,
    len = (src_phrs_len + tgt_phrs_len) // 2
    We only summarize phrase length up to  10.
"""
import argparse

parser = argparse.ArgumentParser('Get distributional info for each sentence')
parser.add_argument('--align', type=str, required=True,
                    help='phrase alignment files')
parser.add_argument('--reserve-length', type=int, required=True,
                    help='reserve phrases with such length for a sentence pair')

args = parser.parse_args()

length_count_dict = {}
sent_count = 0

saveto = args.align + '.length%d' % args.reserve_length

with open(args.align, 'r', encoding='utf-8') as f, open(saveto, 'w') as f_:
    for line in f:
        alignments = line.strip().split()
        reserved_alignments = []
        for align in alignments:
            # import ipdb; ipdb.set_trace()
            m_n, p_q = align.split(':')
            m, n = m_n.split('-')
            p, q = p_q.split('-')
            m, n, p, q = int(m), int(n), int(p), int(q)
            # phrs_len = ((n - m + 1) + (q - p) + 1) // 2
            phrs_len = n - m + 1
            # phrs_len = q - p + 1
            if phrs_len > 10:
                continue
            if phrs_len in length_count_dict:
                length_count_dict[phrs_len] += 1
            else:
                length_count_dict[phrs_len] = 1
            if phrs_len == args.reserve_length:
                reserved_alignments.append(align)
        reserved_alignments = ' '.join(reserved_alignments) + '\n'
        f_.write(reserved_alignments)
        sent_count += 1

# print statistics
lc_list = list(length_count_dict.items())
lc_list = sorted(lc_list, key=lambda t: t[0], reverse=False)
for lc in lc_list:
    l, c = lc
    print('Length %d phrase: %d (total %d)' % (l, c/sent_count, c))



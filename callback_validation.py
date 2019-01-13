"""
Author: Guanlin Li
Date  : Jan. 13 2019
"""

import subprocess
import os
import glob
import time

import argparse

parser = argparse.ArgumentParser("Callback validation script: do BLEU val for each saved ckpts.")
parser.add_argument("--ckpt-path", type=str, required=True,
                    help="folder to hold all ckpts, "
                         "e.g. 'checkpoints/iwslt14.de-en.transformer.word.EXP1'")
parser.add_argument("--input-data", type=str, required=True,
                    help="the data path as same as train.")
parser.add_argument("--EXP", type=str, required=True,
                    help="the id of the experiment, e.g. EXP1.")
parser.add_argument("--TMP", type=str, default="tmp",
                    help="tempory folder to hold inference results.")
parser.add_argument("--corpus", type=str, required=True,
                    help="folder named after the experimental corpus.")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--beam-size", type=int, default=5)
# currently, test both valid and test
# parser.add_argument("--split", type=str, default="valid")
args = parser.parse_args()

# initialize args
TMP = args.TMP
input = args.input_data
data = args.corpus
EXP = args.EXP
splits = ["valid", "test"]
ckpt_folder = args.ckpt_path
batch_size = args.batch_size
beam = args.beam_size
ckpt_log = os.path.join(args.ckpt_path, "validated_ckpt_log.txt")

new_ckpts = []
# create of load validated ckpt log
finished_ckpts = set()
if os.path.exists(ckpt_log):
    with open(ckpt_log, 'r') as f:
        for line in f:
            ckpt_name = line.strip()
            finished_ckpts.add(ckpt_name)
else:
    # only create
    with open(ckpt_log, 'w') as f:
        pass


def filter_ckpts(filenames, finished_ckpts):
    # filter out last, best and already validated ckpts
    new_filenames = []
    for n in filenames:
        if 'checkpoint' not in n:
            continue
        if 'last' in n:
            continue
        if 'best' in n:
            continue
        if n in finished_ckpts:
            continue
        new_filenames.append(n)

    return new_filenames


try_mkdir_cmd="mkdir -p {}/{}/{}".format(TMP, data, EXP)

while True:
    # load validated ckpt log
    finished_ckpts = set()
    with open(ckpt_log, 'r') as f:
        for line in f:
            ckpt_name = line.strip()
            finished_ckpts.add(ckpt_name)
    # sort new_ckpts by date
    filenames = glob.glob("{}/*.pt".format(ckpt_folder))
    filenames.sort(key=os.path.getmtime)
    # new_ckpts = sorted(new_ckpts, reverse=False)
    new_ckpts = filter_ckpts(filenames, finished_ckpts)

    for ckpt in new_ckpts:
        for split in splits:
            ckpt_name = ckpt.split('/')[-1]

            # "tmp/iwslt14.de-en/EXP1/ckpt_name.test.pred"
            infer_save_to = TMP + '/' + data + '/' + EXP + '/' + ckpt_name + '.' + split + '.pred'
            goldn_save_to = TMP + '/' + data + '/' + EXP + '/' + ckpt_name + '.' + split + '.ref'

            # ckpt = os.path.join(ckpt_folder, ckpt_name)

            run_python_cmd = "python generate.py " \
                             "{} " \
                             "--gen-subset {} " \
                             "--path {} " \
                             "--batch-size {} " \
                             "--beam {} " \
                             "--infer-save-to {} " \
                             "--goldn-save-to {} " \
                             "--quiet".format(input, split, ckpt, batch_size, beam,
                                              infer_save_to, goldn_save_to)

            # write to log.txt
            os.system("echo '' | tee -a {}/{}/{}/log.txt".format(TMP, data, EXP))
            os.system("echo 'begin validating ckpt {} on {}' "
                      "| tee -a {}/{}/{}/log.txt".format(ckpt, split, TMP, data, EXP))
            os.system("date | tee -a {}/{}/{}/log.txt".format(TMP, data, EXP))
            # mkdir -p ...
            os.system(try_mkdir_cmd)
            # python generate.py ...
            os.system(run_python_cmd)
            os.system("echo {} | tee -a {}/{}/{}//log.txt".format(ckpt, TMP, data, EXP))
            os.system("./scripts/multi-bleu.perl "
                      "{}/{}/{}/{}.{}.ref < {}/{}/{}/{}.{}.pred "
                      "| tee -a {}/{}/{}/log.txt".format(TMP, data, EXP, ckpt_name, split,
                                                         TMP, data, EXP, ckpt_name, split,
                                                         TMP, data, EXP))
            os.system("echo 'end' | tee -a {}/{}/{}/log.txt".format(TMP, data, EXP))
            os.system("echo '' | tee -a {}/{}/{}/log.txt".format(TMP, data, EXP))

        # add validated ckpt names to 'validated_ckpt_log.txt'
        with open(ckpt_log, 'a+') as f:
            f.write(ckpt + '\n')

    time.sleep(60)

# print("start validate on newly saved checkpoints")
# subprocess.call("bash/iwslt14.de2en.infer.EXP1.sh", shell=True)
# print("end")

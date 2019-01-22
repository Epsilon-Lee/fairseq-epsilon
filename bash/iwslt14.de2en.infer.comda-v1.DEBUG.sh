TMP=tmp
data=iwslt14.de-en
EXP=DEBUG
split=valid
ckpt=checkpoints/iwslt14.de-en/comda-v1.word.DEBUG.run2/checkpoint_best.pt
mkdir -p $TMP/$data
python generate.py \
    data-bin/iwslt14.tokenized.de-en.nobpe.vocabSwitchout.comda-v1 \
    --task translation_comda_xxx \
    --use-xxx \
    --gen-subset $split \
    --path $ckpt \
    --batch-size 32 \
    --beam 5 \
    --infer-save-to $TMP/$data/$EXP.$split.pred \
    --goldn-save-to $TMP/$data/$EXP.$split.ref \
    --quiet \
    --raw-text \
    # | tee tmp/fconv.test.de2en.pred

echo "" | tee -a $TMP/$data/log.txt
date | tee -a $TMP/$data/log.txt
echo $ckpt | tee -a $TMP/$data/log.txt
./scripts/multi-bleu.perl $TMP/$data/$EXP.$split.ref < $TMP/$data/$EXP.$split.pred | tee -a $TMP/$data/log.txt


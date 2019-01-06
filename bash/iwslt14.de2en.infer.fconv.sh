python generate.py \
    data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/fconv/checkpoint_best.pt \
    --batch-size 32 \
    --beam 5 \
    --remove-bpe '@@ ' \
    --infer-save-to tmp/fconv.test.de2en.pred \
    --goldn-save-to tmp/fconv.test.de2en.ref \
    # | tee tmp/fconv.test.de2en.pred

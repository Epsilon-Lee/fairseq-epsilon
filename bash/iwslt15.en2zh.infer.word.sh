python generate.py \
    data-bin/iwslt15.tokenized.en-zh.word \
    --path checkpoints/transformer_nobpe.en-zh.word/checkpoint_best.pt \
    --batch-size 32 \
    --beam 5 \
    --infer-save-to tmp/transformer.en2zh.word.test.pred \
    --goldn-save-to tmp/transformer.en2zh.word.test.ref \
    --gen-subset test \
    # --remove-bpe '@@ ' \
    # | tee tmp/fconv.test.de2en.pred

tmp1=tmp/transformer_nobpe.vocabSwitchout_v4.test.de2en.pred
tmp2=tmp/transformer_nobpe.vocabSwitchout_v4.test.de2en.ref
python generate.py \
    data-bin/iwslt14.tokenized.de-en.nobpe.vocabSwitchout \
    --path checkpoints/transformer_nobpe.vocabSwitchout_v4/checkpoint_best.pt \
    --batch-size 32 \
    --beam 5 \
    --infer-save-to $tmp1 \
    --goldn-save-to $tmp2 \
    --gen-subset test \
    # --remove-bpe '@@ ' \
    # | tee tmp/fconv.test.de2en.pred

./scripts/multi-bleu.perl $tmp2 < $tmp1

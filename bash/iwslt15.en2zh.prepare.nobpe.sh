TEXT=examples/translation/iwslt15/en-zh/final
python preprocess.py \
    --source-lang en \
    --target-lang zh \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --nwordssrc 31098 \
    --nwordstgt 39064 \
    --destdir data-bin/iwslt15.tokenized.en-zh.word.raw \
    --output-format raw

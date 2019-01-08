# prepare train
# TEXT=examples/translation/iwslt15/en-zh/final
# prepare proto
TEXT=examples/translation/iwslt15/en-zh/final/brown/final
python preprocess.py \
    --source-lang en \
    --target-lang zh \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --nwordssrc 50 \
    --nwordstgt 50 \
    --destdir data-bin/iwslt15.tokenized.en-zh.word.proto \
    --output-format raw \
    --padding-factor 1

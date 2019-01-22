TEXT=examples/translation/iwslt14.tokenized.de-en.nobpe
python preprocess.py \
    --source-lang de \
    --target-lang en \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --nwordstgt 22801 \
    --nwordssrc 30001 \
    --destdir data-bin/iwslt14.tokenized.de-en.nobpe.vocabSwitchout.comda-v1 \
    --use-xxx \
    --output-format raw \

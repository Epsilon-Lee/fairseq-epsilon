TEXT=examples/translation/iwslt14.tokenized.de-en.nobpe
python preprocess.py \
    --source-lang de \
    --target-lang en \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --nwordstgt 22800 \
    --nwordssrc 30000 \
    --destdir data-bin/iwslt14.tokenized.de-en.nobpe.vocabSwitchout

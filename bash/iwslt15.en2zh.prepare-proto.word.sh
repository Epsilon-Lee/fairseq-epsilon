# prepare train
# TEXT=examples/translation/iwslt15/en-zh/final
# prepare proto
ALIGN=examples/translation/iwslt15/en-zh/final/fastalign
TEXT=examples/translation/iwslt15/en-zh/final/brown/final
DEST_DIR_PROTO=data-bin/iwslt15.tokenized.en-zh.word.proto
DEST_DIR_COMDA=data-bin/iwslt15.tokenized.en-zh.word.comda splits="train valid test"
trans=en-zh
langs="zh en"
python preprocess.py \
    --source-lang en \
    --target-lang zh \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --nwordssrc 54 \
    --nwordstgt 54 \
    --destdir $DEST_DIR_PROTO \
    --output-format raw \
    --padding-factor 1

# cp corpus proto to comda
for split in $splits
do
    for lang in $langs
    do
        cp $DEST_DIR_PROTO/$split.$trans.$lang $DEST_DIR_COMDA/$split.$trans.$lang.proto
    done
done

# cp proto dict to comda
for lang in $langs
do
    cp $DEST_DIR_PROTO/dict.$lang.txt $DEST_DIR_COMDA/dict.$lang.proto.txt
done

# cp alignment file to comda
cp $ALIGN/bidirection.align $DEST_DIR_COMDA/train.$trans.align

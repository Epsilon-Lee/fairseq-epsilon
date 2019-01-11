corpus=./data-bin/iwslt14.tokenized.de-en.nobpe.vocabSwitchout
arch=transformer_iwslt_de_en
save_dir=checkpoints/iwslt14.de-en.transformer.word.EXP1
mkdir -p $save_dir
CUDA_VISIBLE_DEVICES=0 python train.py \
    $corpus \
    --lr 0.001 \
    --decay-until 8000 \
    --lr-scheduler switchout \
    --optimizer adam \
    --min-lr 0.0 \
    --max-update 100000 \
    --save-interval-updates 2500 \
    --update-freq 3 \
    --clip-norm 25.0 \
    --dropout 0.25 \
    --max-tokens 1500 \
    --arch $arch \
    --save-dir $save_dir \
    --no-progress-bar \
    --log-interval 25 \
    --encoder-embed-dim 320 \
    --encoder-ffn-embed-dim 507 \
    --encoder-attention-heads 5 \
    --encoder-layers 2 \
    --decoder-embed-dim 320 \
    --decoder-ffn-embed-dim 507 \
    --decoder-attention-heads 5 \
    --decoder-layers 2 \

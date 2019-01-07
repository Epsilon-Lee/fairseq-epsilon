corpus=data-bin/iwslt15.tokenized.en-zh.word
arch=transformer_iwslt_de_en
save_dir=checkpoints/iwslt15.en-zh.transformer.word-v2
mkdir -p $save_dir
CUDA_VISIBLE_DEVICES=0 python train.py \
    $corpus \
    --encoder-embed-dim 256 \
    --encoder-ffn-embed-dim 512 \
    --encoder-attention-heads 4 \
    --encoder-layers 3 \
    --decoder-embed-dim 256 \
    --decoder-ffn-embed-dim 512 \
    --decoder-attention-heads 4 \
    --decoder-layers 3 \
    --update-freq 4 \
    --optimizer adam \
    --lr 0.001 \
    --lr-scheduler switchout \
    --lr-shrink 0.75 \
    --clip-norm 25 \
    --dropout 0.25 \
    --max-tokens 50 \
    --arch $arch \
    --save-dir $save_dir \
    --no-progress-bar \
    --log-interval 25 \
    --decay-until 4000 \
    --da-strategy switchout \
    # --force-anneal 0 \

corpus=./data-bin/iwslt14.tokenized.de-en.nobpe.vocabSwitchout.comda-v1.debug
arch=transformer_iwslt_de_en
save_dir=checkpoints/iwslt14.de-en/baseline.transformer.word.debug
mkdir -p $save_dir
CUDA_VISIBLE_DEVICES=0 python train.py \
    $corpus \
    --raw-text \
    --encoder-embed-dim 256 \
    --encoder-ffn-embed-dim 512 \
    --encoder-attention-heads 4 \
    --encoder-layers 2 \
    --decoder-embed-dim 256 \
    --decoder-ffn-embed-dim 512 \
    --decoder-attention-heads 4 \
    --decoder-layers 2 \
    --update-freq 3 \
    --optimizer adam \
    --lr 0.001 \
    --lr-scheduler switchout \
    --lr-shrink 0.75 \
    --clip-norm 25 \
    --dropout 0.25 \
    --max-tokens 2000 \
    --arch $arch \
    --save-dir $save_dir \
    --no-progress-bar \
    --log-interval 1 \
    --decay-until 4000 \
    # --da-strategy switchout \
    # --force-anneal 0 \

corpus=./data-bin/iwslt14.tokenized.de-en.nobpe.vocabSwitchout.comda-v1
arch=transformer_iwslt_de_en
save_dir=checkpoints/iwslt14.de-en/comda-v1.word.DEBUG.run2
mkdir -p $save_dir
CUDA_VISIBLE_DEVICES=0 python train.py \
    $corpus \
    --raw-text \
    --task translation_comda_xxx \
    --use-xxx \
    --da-strategy prototype \
    --coefficient 0.9975 \
    --criterion cross_entropy_with_encinv \
    --lr 0.001 \
    --decay-until 8000 \
    --lr-scheduler switchout \
    --lr-shrink 0.1 \
    --optimizer adam \
    --min-lr 0.0 \
    --max-epoch 20 \
    --update-freq 3 \
    --clip-norm 25.0 \
    --dropout 0.25 \
    --max-tokens 1500 \
    --update-freq 3 \
    --arch $arch \
    --save-dir $save_dir \
    --no-progress-bar \
    --log-interval 15 \
    --encoder-embed-dim 256 \
    --encoder-ffn-embed-dim 512 \
    --encoder-attention-heads 4 \
    --encoder-layers 2 \
    --decoder-embed-dim 256 \
    --decoder-ffn-embed-dim 512 \
    --decoder-attention-heads 4 \
    --decoder-layers 2 \

corpus=data-bin/iwslt14.tokenized.de-en
arch=transformer_iwslt_de_en
save_dir=checkpoints/transformer
mkdir -p $save_dir
CUDA_VISIBLE_DEVICES=0 python train.py \
    $corpus \
    --lr 0.25 \
    --update-freq 4 \
    --clip-norm 0.1 \
    --dropout 0.2 \
    --max-tokens 1000 \
    --arch $arch \
    --save-dir $save_dir \
    --no-progress-bar \
    --log-interval 25 \

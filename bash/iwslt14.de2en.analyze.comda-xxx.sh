corpus=./data-bin/iwslt14.tokenized.de-en.nobpe.vocabSwitchout.comda-xxx.analysis
arch=transformer_iwslt_de_en
save_dir=checkpoints/iwslt14.de-en/comda-v1.word.DEBUG
mkdir -p $save_dir
CUDA_VISIBLE_DEVICES=0 python analyze.py \
    $corpus \
    --restore-file checkpoint_best.pt \
    --raw-text \
    --task perturb_analysis \
    --use-xxx \
    --da-strategy prototype \
    --coefficient 0.9975 \
    --criterion perturb_divergence \
    --lr 0.001 \
    --decay-until 8000 \
    --lr-scheduler switchout \
    --lr-shrink 0.1 \
    --optimizer adam \
    --min-lr 0.0 \
    --max-epoch 1 \
    --update-freq 1 \
    --clip-norm 25.0 \
    --dropout 0.25 \
    --max-sentences 1 \
    --arch $arch \
    --save-dir $save_dir \
    --no-progress-bar \
    --log-interval 1 \
    --encoder-embed-dim 256 \
    --encoder-ffn-embed-dim 512 \
    --encoder-attention-heads 4 \
    --encoder-layers 2 \
    --decoder-embed-dim 256 \
    --decoder-ffn-embed-dim 512 \
    --decoder-attention-heads 4 \
    --decoder-layers 2 \

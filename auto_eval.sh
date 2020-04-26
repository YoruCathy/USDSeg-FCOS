for((i=12;i>=12;i--));  
do
CUDA_VISIBLE_DEVICES=7 python tools/test.py \
configs/usdseg/usd_r50_cosine.py \
work_dirs/usd_r50_cosine/epoch_${i}.pth \
--eval bbox segm
done

for((i=4;i>=1;i--));  
do
CUDA_VISIBLE_DEVICES=7 python tools/test.py configs/usdseg/usd_r50_caffe_fpn_gn_1x_4gpu.py work_dirs/usd_fcos_r50/epoch_${i}.pth --eval bbox segm
done
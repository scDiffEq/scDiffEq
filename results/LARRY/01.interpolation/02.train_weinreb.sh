# example script for training 5 seeds of 2 layer 400 unit models
# on Weinreb et al. dataset on only lineage tracing data with ground truth proliferation rates
# for interpolation task

train_dt=0.1
train_sd=0.1
train_tau=1e-6
train_batch=0.1
train_clip=0.1 
train_lr=0.005
train_epochs=2500
pretrain_epochs=500
k_dim=400 
layers=2
save=100

data_path=../data/Weinreb2020_impute.pt
weight_path=../data/Weinreb2020_growth-d26.pt
out_dir=../experiments/weinreb-interpolate

device=0
for seed in {1,2,3,4,5}
do
  python weinreb.py --task interpolate \
                    --train \
                    --data_path ${data_path}   \
                    --weight_path ${weight_path}  \
                    --out_dir ${out_dir}  \
                    --train_batch ${train_batch}  \
                    --train_clip ${train_clip}  \
                    --train_lr ${train_lr}  \
                    --train_dt ${train_dt}  \
                    --train_sd ${train_sd}  \
                    --train_tau ${train_tau}  \
                    --pretrain_epochs ${pretrain_epochs}  \
                    --train_epochs ${train_epochs}  \
                    --k_dim ${k_dim}  \
                    --layers ${layers}  \
                    --save ${save}  \
                    --seed ${seed}  \
                    --device ${device}
done

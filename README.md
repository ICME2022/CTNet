# R-S-R(Spatio-Temporal Self-Supervision with Attention for Action Recognition)
Overview of Our R-S-R
![Overview of Our R-S-R](https://github.com/ZhangHerman/R-S-R/blob/main/R-S-R.PNG)
#### \R-S-R\selfsupervision\MirrorSSB\RBRF_2.py
Overview of Our RBRF Algorithm (The Case Where The Current Batch is Selected)
![Overview of Our RBRF Algorithm (The Case Where The Current Batch is Selected)](https://github.com/ZhangHerman/R-S-R/blob/main/RBRF.PNG)
###### RBRF Algorithm with 2 kinds of pseudo-labels for the final classification
#### \R-S-R\selfsupervision\MirrorSSB\RBRF.py
###### RBRF Algorithm with 5 kinds of pseudo-labels for the final classification
#### \R-S-R\selfsupervision\shuffle\reverse_random_batch_channel.py
###### RBRC Algorithm
#### \R-S-R\ops\Strongly_constrained_self_attention.py
Overview of Our Spatio-Temporal Hybrid Modeling branch
![Overview of Our Spatio-Temporal Hybrid Modeling branch](https://github.com/ZhangHerman/R-S-R/blob/main/STHM.PNG)
###### Spatio-Temporal Hybrid Modeling branch

# Training and Testing

### HMDB51 and UCF101 Datasets

python -m torch.distributed.launch --master_port 10086  --nproc_per_node=1 main.py Dataset_Name RGB --arch resnet50  --num_segments 16 --gd 20 --lr 0.00015 --lr_steps 25 35 --epochs 50 --batch-size 4 --dropout 0.8 --consensus_type=avg --eval-freq=1 -j 4 --tune_from='your_path/Kinetics.pth.tar'

### Something-Something V1 Dataset

python -m torch.distributed.launch --master_port 10086  --nproc_per_node=1 main.py Dataset_Name RGB --arch resnet50  --num_segments 16 --gd 20 --lr 0.00015 --lr_steps 25 35 --epochs 50 --batch-size 4 --dropout 0.8 --consensus_type=avg --eval-freq=1 -j 4

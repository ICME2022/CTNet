# STTNet(Spatio-Temporal Self-Supervision enhanced Transformer Networks for Action Recognition)
Overview of Our STTNet
![Overview of Our CTNet](https://github.com/ICME2022/CTNet/blob/main/CTNet.PNG)
#### \R-S-R\selfsupervision\MirrorSSB\RBRF_2.py
Overview of Our RBRF Algorithm in Self-Supervised Spatial Representation Learning Module(The Case Where The Current Batch is Selected)
![Overview of Our RBRF Algorithm in Self-Supervised Spatial Representation Learning Module(The Case Where The Current Batch is Selected)](https://github.com/ICME2022/CTNet/blob/main/RBRF.PNG)
###### RBRF Algorithm with 2 kinds of pseudo-labels for the final classification
#### \R-S-R\selfsupervision\MirrorSSB\RBRF.py
###### RBRF Algorithm with 5 kinds of pseudo-labels for the final classification
#### \R-S-R\selfsupervision\shuffle\reverse_random_batch_channel.py
###### RBRC Algorithm
#### \R-S-R\ops\Strongly_constrained_self_attention.py
Overview of Transformer based Spatio-temporal Aggregator
![Overview of Our Spatio-Temporal Contextual Transformer Module](https://github.com/ICME2022/CTNet/blob/main/Contextual%20Transformer.PNG)



# Training and Testing

### HMDB51 and UCF101 Datasets

python -m torch.distributed.launch --master_port 17686  --nproc_per_node=1 main.py Dataset_Name RGB --arch resnet50  --num_segments 16 --gd 20 --lr 0.00015 --lr_steps 25 35 --epochs 50 --batch-size 4 --dropout 0.8 --consensus_type=avg --eval-freq=1 -j 4 --tune_from='your_path/Kinetics.pth.tar'

### Something-Something V1 Dataset

python -m torch.distributed.launch --master_port 19486  --nproc_per_node=1 main.py something RGB --arch resnet50  --num_segments 16 --gd 20 --lr 0.0013 --lr_steps 35 45 55 --epochs 65 --batch-size 8 --dropout 0.8 --consensus_type=avg --eval-freq=1 -j 4

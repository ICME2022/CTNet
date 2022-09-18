#NEWS!

2022.3
Our paper is accepted by ICME2022 as ORAL!!!

# STTNet(Spatio-Temporal Self-Supervision enhanced Transformer Networks for Action Recognition)
Overview of Our STTNet
![Overview of Our CTNet](https://github.com/ICME2022/STTNet/blob/main/STTNet_main.PNG)
#### \R-S-R\selfsupervision\MirrorSSB\RBRF_2.py
Overview of Our RBRF Algorithm in Self-Supervised Spatial Representation Learning Module(The Case Where The Current Batch is Selected)
![Overview of Our RBRF Algorithm in Self-Supervised Spatial Representation Learning Module(The Case Where The Current Batch is Selected)](https://github.com/ICME2022/CTNet/blob/main/RBRF.PNG)
###### RBRF Algorithm with 2 kinds of pseudo-labels for the final classification
#### \R-S-R\selfsupervision\MirrorSSB\RBRF.py
###### RBRF Algorithm with 5 kinds of pseudo-labels for the final classification
#### \R-S-R\selfsupervision\shuffle\reverse_random_batch_channel.py
###### RBRC Algorithm
#### \R-S-R\ops\Strongly_constrained_self_attention.py



# Data Preparation
We have successfully trained STTNet on [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/), [UCF101](https://www.crcv.ucf.edu/data/UCF101.php), [Something-Something-V1](https://20bn.com/datasets/something-something/v1) with this codebase (Pre-trained Models are on https://drive.google.com/drive/folders/18JslcTMTrjXJPn-I7Dnof1GKj4J6cOE1).

## The processing of HMDB51, UCF101 and Something-Something-V1 can be summarized into 3 steps:
i.Extract frames from videos(you can use ffmpeg to get frames from video)

ii.Generate annotations needed for dataloader ("<path_to_frames> <frames_num> <video_class>" in annotations) The annotation usually includes train.txt and val.txt. 
The format of *.txt file is like:
```
dataset_root/frames/video_1 num_frames label_1
dataset_root/frames/video_2 num_frames label_2
dataset_root/frames/video_3 num_frames label_3
...
dataset_root/frames/video_N num_frames label_N
```

iii.Add the information to ops/dataset_configs.py

# Prerequisites
The code or our STTNet is built with following libraries:

 · Python 3.6 or higher
 
 · PyTorch 1.4 or higher
 
 · Torchvision
 
 · TensorboardX
 
 ```pip install tensorboardX```
 
 · tqdm
 
 ```pip install tqdm```
 
 · scikit-learn
 
 ```pip install -U scikit-learn```
 
 · ffmpeg
 
 ```
 conda config --add channels conda-forge
 conda install ffmpeg
 pip install ffmpy
 ```
 
 · decord
 
 ```pip install decord```

# We recommend that you train STTNet on Tesla V100 GPU

## Training
### HMDB51 and UCF101 Datasets

```
python -m torch.distributed.launch --master_port 17686  --nproc_per_node=1 main.py Dataset_Name RGB --arch resnet50  --num_segments 16 --gd 20 --lr 0.00015 --lr_steps 25 35 --epochs 50 --batch-size 4 --dropout 0.8 --consensus_type=avg --eval-freq=1 -j 4 --tune_from='your_path/Kinetics.pth.tar'
```

### Something-Something V1 Dataset

```
python -m torch.distributed.launch --master_port 19486  --nproc_per_node=1 main.py something RGB --arch resnet50  --num_segments 16 --gd 20 --lr 0.0013 --lr_steps 35 45 55 --epochs 65 --batch-size 8 --dropout 0.8 --consensus_type=avg --eval-freq=1 -j 4
```

## Testing

```
CUDA_VISIBLE_DEVICES=0 python3 test_models_center_crop.py something \
--archs='resnet50' --weights <your_checkpoint_path>  --test_segments=16  \
--test_crops=1 --batch_size=8  --gpus 0 --output_dir <your_pkl_path> -j 4 --clip_index=0
```
 

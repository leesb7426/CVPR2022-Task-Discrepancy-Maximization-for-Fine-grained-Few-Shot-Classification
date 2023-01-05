# Task-Discrepancy-Maximization-for-Fine-grained-Few-Shot-Classification
Official PyTorch Repository of "[Task Discrepancy Maximization for Fine-grained Few-Shot Classification (CVPR 2022 Oral Paper)](https://openaccess.thecvf.com/content/CVPR2022/html/Lee_Task_Discrepancy_Maximization_for_Fine-Grained_Few-Shot_Classification_CVPR_2022_paper.html)"

<!--
  Title: 
  Description: This is the code for .
  Author: 
  -->

<figure style="text-align: center;">
<!--   <figcaption style="text-align: center;"><b>Overall Framework</b></figcaption> -->
    <img src=figures/overall.png width="95%"> 
</figure>

<!-- <figure style="text-align: center;">
  <p align="center">
  <figcaption style="text-align: center;"><b>Support Attention Module</b></figcaption>
    <img src=figures/SAM.png width="45%"> 
  </p>
</figure>

<figure style="text-align: center;">
  <p align="center">
  <figcaption style="text-align: center;"><b>Query Attention Module</b></figcaption>
    <img src=figures/QAM.png width="45%"> 
  </p>
</figure> -->


<!-- 
<p align="center">
    <img src=figures/SAM.png width="45%"> 
    <img src=figures/QAM.png width="45%"> 
  <figcaption align = "center"><b>SAM     QAM</b></figcaption>
</p> -->


## Bug Fix
[2023/01/03] We omitted the random loss, which prevents overfitting, in training. Please add "--noise" to the training code. Sorry for the confusion.

## Data Preparation

<!-- Please download the dataset before you run the code.

CUB_200_2011: [CUB_200_2011 download link](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)

FGVC-Aircraft: [FGVC-Aircraft download link](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)

iNaturalist2017 : [[iNaturalist2017 Dataset Page](https://github.com/visipedia/inat_comp/tree/master/2017), [iNaturalist2017 Download Data](https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz), [iNaturalist2017 Download Annotations](https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_2017_bboxes.zip)\]

Stanford-Cars : [Stanford-Cars homepage](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)

Stanford-Dogs : [Stanford-Dogs homepage](http://vision.stanford.edu/aditya86/ImageNetDogs/)

Oxford-Pets : [Oxford-Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/)
 -->
 The following datasets are used in our paper:

CUB_200_2011: [Dataset Page](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

FGVC-Aircraft: [Dataset Page](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)

iNaturalist2017 : [Dataset Page](https://github.com/visipedia/inat_comp/tree/master/2017)

Stanford-Cars : [Dataset Page](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)

Stanford-Dogs : [Dataset Page](http://vision.stanford.edu/aditya86/ImageNetDogs/)

Oxford-Pets : [Dataset Page](https://www.robots.ox.ac.uk/~vgg/data/pets/)

Please proceed with the setting up data by referring to [FRN Github](http://github.com/Tsingularity/FRN#setting-up-data).


## Pretrained Weights

Here, all the pretrained weights with ProtoNet and FRN are publicized.
[Pretrained weights](https://drive.google.com/drive/folders/15IZtDiMzJT_v49b5AKEdK_2QnBLNegdO?usp=sharing)
<!--  Method | Dataset | 1-shot | 5-shot | Model file
 -- | -- | -- | -- | --
  |  |  |  | [model]()
 |  |  |  | [model]()
  |  |   | | [model]() -->

## Usage

### Requirement
All the requirements to run the code are in requirements.txt
You can download requirements by running below script.
```
pip install -r requirements.txt
```

<!-- ### Dataset directory
Change the data_path in config.yml.
```
dataset_path: #your_dataset_directory
```
 -->
### Evaluation
To evaluate the code with pretrained weights, we provide an example script below.

Test the ProtoNet 1-shot in CUB_cropped with Conv-4.  
(pretrained_weight: /Proto/CUB_fewshot_cropped/TDM/Conv4-1shot/model_Conv-4.pth)
```
python3 test.py --train_way 5 --train_shot 1 --gpu_num 1 --model Proto --dataset cub_cropped --TDM
```
Test the ProtoNet 5-shot in CUB_cropped with Conv-4.  
(pretrained_weight: /Proto/CUB_fewshot_cropped/TDM/Conv4-5shot/model_Conv-4.pth)
```
python3 test.py --train_way 5 --train_shot 5 --gpu_num 1 --model Proto --dataset cub_cropped --TDM
```
Test the FRN 1-shot and 5-shot in CUB_cropped with Conv-4.  
(pretrained_weight: /FRN/CUB_fewshot_cropped/TDM/Conv4-5shot/model_Conv-4.pth)
```
python3 test.py --train_way 5 --train_shot 5 --gpu_num 1 --model FRN --dataset cub_cropped --TDM
```

### Train
We provide scripts for training. Other shell scripts are in the scripts directory. 
```
python3 train.py --model Proto --dataset aircraft --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 3 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 30 --train_shot 1 --train_transform_type 0 --test_shot 1 --pre --gpu_num 1
```

## Results 
As we mentioned in our supplementary, we found that CTX and DSN show better performances in 1-shot when they are evaluated by models trained with 5-shot episodes.
Therefore, we trained all models with 5-shot episodes except ProtoNet because it shows performance degradation.

### CUB_cropped
<figure style="text-align: center;">
    <img src=figures/CUB_cropped.PNG width="50%"> 
</figure>

### CUB_raw
<figure style="text-align: center;">
    <img src=figures/CUB_raw.PNG width="50%"> 
</figure>

### Aircraft
<figure style="text-align: center;">
    <img src=figures/Aircraft.PNG width="50%"> 
</figure>

### meta-iNat & tiered meta-iNat
<figure style="text-align: center;">
    <img src=figures/iNat.PNG width="50%"> 
</figure>

## Citation
If you find TDM helpful for your works, please consider citing:
```
@InProceedings{Lee_2022_CVPR,
    author    = {Lee, SuBeen and Moon, WonJun and Heo, Jae-Pil},
    title     = {Task Discrepancy Maximization for Fine-Grained Few-Shot Classification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {5331-5340}
}
```

## Contact
If there are any questions, please feel free to contact with the authors:  SuBeen Lee (leesb7426@gmail.com) WonJun Moon (wjun0830@gmail.com). Thank you.

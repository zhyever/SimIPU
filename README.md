# SimIPU

> **SimIPU: Simple 2D Image and 3D Point Cloud Unsupervised Pre-Training for Spatial-Aware Visual Representations**
>
> Zhenyu Li, Zehui Chen, Ang Li, Liangji Fang, Qinhong Jiang, Xianming Liu, Junjun Jiang, Bolei Zhou, Hang Zhao
>
> [AAAI 2021 (arXiv pdf)](https://arxiv.org/abs/2112.04680)

## Notice
- Redundancy version of SimIPU. Main codes are in SimIPU/project_cl.
- You can find codes of MonoDepth [here](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/simipu). We provide detailed configs and results, even in an indoor environment depth dataset, which demonstrates the generalization of SimIPU. Since we enhance the depth framework, model performances are stronger than the ones presented in our paper.

## Usage

### Installation

This repo is tested on python=3.7, cuda=10.1, pytorch=1.6.0, [mmcv-full=1.3.4](https://github.com/open-mmlab/mmcv), 
[mmdetection=2.11.0](https://github.com/open-mmlab/mmdetection), [mmsegmentation=0.13.0](https://github.com/open-mmlab/mmsegmentation) and 
[mmdetection3D=0.13.0](https://github.com/open-mmlab/mmdetection3d).

Note: since mmdetection and mmdetection3D have made huge compatibility change in their latest versions, their latest version is not compatible with this repo.
Make sure you install the correct version. 

Follow instructions below to install:

- **Create a conda environment**

```
conda create -n simipu python=3.7
conda activate monocon
git clone https://github.com/zhyever/SimIPU.git
cd SimIPU
```

- **Install Pytorch 1.6.0**

```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

- **Install mmcv-full=1.3.4**

```
pip install mmcv-full==1.3.4 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
```

- **Install mmdetection=2.11.0**

```
git clone https://github.com/open-mmlab/mmdetection.git
cd ./mmdetection
git checkout v2.11.0
pip install -r requirements/build.txt
pip install -v -e .
cd ..
```

- **Install mmsegmentation=0.13.0**

```
pip install mmsegmentation==0.13.0
```

- **Build SimIPU**

```
# remember you have "cd SimIPU"
pip install -v -e .
```

- **Others**
Maybe there will be notice that there is no required future package after build SimIPU. Install it via conda.

```
conda install future
```

### Data Preparation

Download [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize data 
 following the [official instructions](https://mmdetection3d.readthedocs.io/en/latest/)
  in mmdetection3D. Then generate data by running:
  
```
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

If you would like to run experiments on Mono3D Nus, you should follow the [official instructions](https://mmdetection3d.readthedocs.io/en/latest/) to prepare the NuScenes dataset.

For Waymo pre-training, we have no plan to release corresponding data-preparing scripts for a short time. Some of the scripts are presented in project_cl/tools/. I just have no effort or resources to reproduce the Waymo pre-training process. Since we provide how to prepare the Waymo dataset in our paper, if you have a problem to achieve it, feel free to contact me and I would like to help you. 

### Pre-training on KITTI
```
bash tools/dist_train.sh project_cl/configs/simipu/simipu_kitti.py 8 --work-dir work_dir/your/work/dir
```

### Downstream Evaluation
#### 1. Camera-lidar fusion based 3D object detection on kitti dataset. 
Remember to change the pre-trained model via changing the value of key `load_from` in the config.
```
bash tools/dist_train.sh project_cl/configs/kitti_det3d/moca_r50_kitti.py 8 --work-dir work_dir/your/work/dir
```
#### 2. Monocular 3D object detection on Nuscenes dataset. 
Remember to change the pre-trained model via changing the value of key `load_from` in the config. Before training, you also need align the key name in `checkpoint['state_dict']`. See `project_cl/tools/convert_pretrain_imgbackbone.py` for details.
```
bash tools/dist_train.sh project_cl/configs/fcos3d_mono3d/fcos3d_r50_nus.py 8 --work-dir work_dir/your/work/dir
```
#### 2. Monocular Depth Estimation on KITTI/NYU dataset. 
See [Depth-Estimation-Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/simipu).


## Pre-trained Model and Results

We provide pre-trained models.
As default, the "Full Waymo or Waymo" presents Waymo dataset with `load_interval=5`. We use discrete frames to ensure training variety. Previous experiments indicate model improvement with `load_interval=1` is slight. So actually, 1/10 Waymo means 1/5 (`load_interval=5`) times 1/10 (use first 1/10 scene data) = 1/50 Waymo data.
|   |Dataset |Model  |
|:-:| :----: | :----:|
|SimIPU|KITTI|[link](https://github.com/zhyever/SimIPU/releases/download/initial-release/SimIPU_kitti_50e.pth)|
|SimIPU|Waymo|[link](https://github.com/zhyever/SimIPU/releases/download/initial-release/SimIPU_waymo.pth)|
|SimIPU|ImageNet Sup + Waymo SimIPU|[link](https://github.com/zhyever/SimIPU/releases/download/double-finetune/SimIPU_imagesup_waymo_double_finetune.pth)|


Fusion-based 3D object detection results.
|         | AP40@Easy | AP40@Mod. | AP40@Hard | Link      |
| :-------: | :---------: |:-----------:|:-----------:|:-----------:|
| Moca    | 81.32     | 70.88     | 66.19     |  [Log](https://github.com/zhyever/SimIPU/blob/main/resources/logs/moca_simipu_kitti.txt) |

Monocular 3D object detection results.
|           | Pre-train  | mAP         | Link      |
| :-------: | :---------:|:-----------:|:-----------:|
| Fcos3D    | Scratch    | 17.9        |  [Log](https://github.com/zhyever/SimIPU/blob/main/resources/logs/fcos3d_scratch_nus.txt) |
| Fcos3D    | 1/10 Waymo SimIPU   | 20.3        |  [Log](https://github.com/zhyever/SimIPU/blob/main/resources/logs/fcos3d_simipu_nus_abl_oneten.txt) |
| Fcos3D    | 1/5 Waymo SimIPU   | 22.5        |  [Log](https://github.com/zhyever/SimIPU/blob/main/resources/logs/fcos3d_simipu_nus_abl_onefive.txt) |
| Fcos3D    | 1/2 Waymo SimIPU  | 24.7        |  [Log](https://github.com/zhyever/SimIPU/blob/main/resources/logs/fcos3d_scratch_nus.txt) |
| Fcos3D    | Full Waymo SimIPU   | 26.2        |  Log |
| Fcos3D    | ImageNet Sup    | 27.7        |  [Log](https://github.com/zhyever/SimIPU/blob/main/resources/logs/fcos3d_imgnet_nus.txt) |
| Fcos3D    | ImageNet Sup + Full Waymo SimIPU   | 28.4    |  Log |




</center>

## Citation
If you find our work useful for your research, please consider citing the paper
```
@article{li2021simipu,
  title={SimIPU: Simple 2D Image and 3D Point Cloud Unsupervised Pre-Training for Spatial-Aware Visual Representations},
  author={Li, Zhenyu and Chen, Zehui and Li, Ang and Fang, Liangji and Jiang, Qinhong and Liu, Xianming and Jiang, Junjun and Zhou, Bolei and Zhao, Hang},
  journal={arXiv preprint arXiv:2112.04680},
  year={2021}
}
```


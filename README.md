<p align="center">
<img src="images/logo.png" height="500" align="center">
</p>

# EgoObjects

Official pytorch implementation of the ICCV 23' paper

**[EgoObjects: A Large-Scale Egocentric Dataset for Fine-Grained Object Understanding](https://arxiv.org/abs/2309.08816)**

[Chenchen Zhu](https://sites.google.com/andrew.cmu.edu/zcckernel), [Fanyi Xiao](https://fanyix.cs.ucdavis.edu/), [Andres Alvarado](https://www.linkedin.com/in/josecarlos12/), [Yasmine Babaei](https://www.linkedin.com/in/yasminebabaei/), [Jiabo Hu](https://www.linkedin.com/in/jiabo-hu-1321b1121/), [Hichem El-Mohri](https://www.linkedin.com/in/hichem-elmohri/), [Sean Chang Culatana](https://ai.meta.com/people/sean-chang-culatana/), [Roshan Sumbaly](https://www.linkedin.com/in/rsumbaly/), [Zhicheng Yan](https://sites.google.com/view/zhicheng-yan)

**Meta AI**

<a href='https://github.com/facebookresearch/EgoObjects'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2309.08816'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

EgoObjects is a large-scale egocentric dataset for fine-grained object understanding, which features videos captured by various wearable devices at worldwide locations, objects from a diverse set of categories commonly seen in indoor environments, and videos of the same object instance captured under diverse conditions. The dataset supports both the conventional category-level as well as the novel instance-level object detection task.

<p align="center">
<img src="images/sample_images.png"/>
</p>

<p align="center">
  <img src="images/taxonomy.png" width="500" height="550">
</p>


## EgoObjects v1.0

For this release, we have annotated 114K frames (79K train, 5.7K val, 29.5K test) sampled from 9K+ videos collected by 250 participants across the world. A total of 14.4K unique object instances from 368 categories are annotated. Among them, there are 1.3K main object instances from 206 categories and 13.1K secondary object instances (i.e., objects accompanying the main object) from 353 categories. On average, each image is annotated with 5.6 instances from 4.8 categories, and each object instance appears in 44.8 images, which ensures diverse viewing directions for the object. 

## Dataset downloading

Release v1.0 is publicly available from this [page](https://ai.meta.com/datasets/egoobjects-downloads). Images (~40G) can be downloaded from file `EgoObjectsV1_images.zip`. Unified annotations for category and instance level object detection can be downloaded from files including `EgoObjectsV1_unified_train.json`, `EgoObjectsV1_unified_eval.json`, and `EgoObjectsV1_unified_metadata.json`. They can be placed under `$EgoObjects_ROOT/data/`. We follow the same data format as [LVIS](https://www.lvisdataset.org/dataset) with EgoObjects specific changes.

## Setup

### Requirements
- Linux with Python ≥ 3.8
- PyTorch ≥ 1.8.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

### Example conda environment setup
```bash
conda create --name egoobjects python=3.9
conda activate egoobjects
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# under your working directory
git clone https://github.com/facebookresearch/EgoObjects.git
cd EgoObjects
```

If setup correctly, run our evaluation example code to get mock results for category and instance level detection tasks:
```bash
python example.py
```

## Timeline
- 23' Sep 6, EgoObjects v1.0, including both data and annotations, is open sourced.
- 23' March, EgoObjects v1.0 is covered by <a href="https://research.facebook.com/blog/2023/3/egoobjects-large-scale-egocentric-dataset-for-category-and-instance-level-object-understanding/">Meta AI</a>
- 22' June, an earlier version of EgoObjects dataset is adopted by the <a href="https://sites.google.com/view/clvision2022/challenge"> Continual Learning Challenge in the 3rd CLVISION Workshop</a> at CVPR.
- 22' March, EgoObjects is first introduced by <a href="https://ai.meta.com/blog/advancing-first-person-perception-with-2022-ego4d-challenge/#ego-objects">Meta AI Blog</a>

## EgoObjects ICCV 23' Poster

<p align="center">
<img src="images/ICCV2023_poster_EgoObjects.jpg" height="500" align="center">
</p>

## Citing EgoObjects
Paper link: [[`arXiv`](https://arxiv.org/abs/2309.08816)]

If you find this code/data useful in your research then please cite our paper:
```
@inproceedings{zhu2023egoobjects,
  title={EgoObjects: A Large-Scale Egocentric Dataset for Fine-Grained Object Understanding},
  author={Zhu, Chenchen and Xiao, Fanyi and Alvarado, Andrés and Babaei, Yasmine and Hu, Jiabo and El-Mohri, Hichem and Chang, Sean and Sumbaly, Roshan and Yan, Zhicheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

## Credit
The code is a re-write of PythonAPI for [LVIS](https://github.com/lvis-dataset/lvis-api).
The core functionality is the same with EgoObjects specific changes.

## License
EgoObjects is licensed under the [MIT License](LICENSE).

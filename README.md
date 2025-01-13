<div align="center">

# OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal

[![ECCV](https://img.shields.io/badge/ECCV%202024-Accepted-informational.svg)](https://eccv.ecva.net/virtual/2024/poster/1048)
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2408.11480-b31b1b.svg)](https://arxiv.org/abs/2408.11480)
![Visitor](https://visitor-badge.laobi.icu/badge?page_id=QMoQ/OAPT)

[Qiao Mo](), [Yukang Ding](), [Jinhua Hao](https://eric-hao.github.io/), [Qiang Zhu](), [Ming Sun](), [Chao Zhou](), [Feiyu Chen](), [Shuyuan Zhu]()

</div>

> Official implementation of OAPT in ECCV2024, which is a transformer-based network designed for double (or multiple) compressed image restoration.

### Architecture
![architecture](pics/pipeline.png)

### Pattern clustering & inv operation 
![pattern clustering](pics/patternclustering.png)

### Experimental results on gray double JPEG images
![results](pics/gray_results.png)

### Visual results
![gray visual results](pics/visuals.png)


## Training details
All the weights are put in [Baidu Netdisk](https://pan.baidu.com/s/1CAXtSt9oEcBHp8zqCBnrzg?pwd=hm52) and [Gdrive](https://drive.google.com/drive/folders/1yZcczbsVdxmsocQMaC9oFcvt6J553D_1?usp=sharing)

| Model(Gray)                                      | Params(M) | Multi-Adds(G) | TrainingSets |           Pretrain model            | iterations |
| :----------------------------------------------: | :-------: | :-----------: | :----------: | :---------------------------------: | :--------: |
| [SwinIR](https://github.com/JingyunLiang/SwinIR) |   11.49   |    293.42     |     DF2K     | 006_CAR_DFWB_s126w7_SwinIR-M_jpeg10 |    200k    |
| [HAT-S](https://github.com/XPixelGroup/HAT)      |   9.24    |    227.14     |     DF2K     |             HAT-S_SRx2              |    800k    |
| [ART](https://github.com/gladzhang/ART)          |   16.14   |    415.51     |     DF2K     |             CAR_ART_q10             |    200k    |
| [OAPT](https://arxiv.org/abs/2408.11480)         |   12.96   |    293.60     |     DF2K     | 006_CAR_DFWB_s126w7_SwinIR-M_jpeg10 |    200k    |

### Setup
The version of PyTorch we used is 1.7.0. Please ensure you have the correct versions of all dependencies by installing from the `requirements.txt` file.
```
pip install -r requirements.txt
python setup.py develop
```

### Train
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=73 oapt/train.py -opt options/Gray/train/train_oapt.yml --launcher pytorch
```

### Test
```
CUDA_VISIBLE_DEVICES=0 python oapt/test.py -opt ./options/Gray/test/test_oapt.yml
```

## Acknowledgement
This project is mainly based on [SwinIR](https://github.com/JingyunLiang/SwinIR) and [HAT](https://github.com/XPixelGroup/HAT).

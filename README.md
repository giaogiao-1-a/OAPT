# OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal 
[Qiao Mo](), [Yukang Ding](), [Jinhua Hao](), [Qiang Zhu](), [Ming Sun](), [Chao Zhou](), [Feiyu Chen](), [Shuyuan Zhu]()

Kuaishou Techonology, UESTC

>Official implement of OAPT in ECCV2024, which is a transformer-based network deigned for double (or multiple) compressed image restoration.

[Paper Link](https://arxiv.org/abs/2408.11480)

---



# TODO List
- [ ] Submit to ARXIV
- [ ] Release main codes of models
- [ ] Release test codes
- [ ] Release training codes


This project is mainly based on [swinir](https://github.com/JingyunLiang/SwinIR) and [hat](https://github.com/XPixelGroup/HAT).

---
### Network Structure
<img src="./pics/gray_visuals.pdf" width="600"/>


### Training details
| Model(Gray) | Params(M) | Multi-Adds(G) | TrainingSets | Pretrain model | iterations |
|--------|:---------:|:---------:|:---------:|:---------:|:---------:|
| [SwinIR](https://github.com/JingyunLiang/SwinIR) |   11.49    | 293.42 | DF2K | 006_CAR_DFWB_s126w7_SwinIR-M_jpeg10 | 200k |
| [HAT-S](https://github.com/XPixelGroup/HAT) |   9.24    | 227.14 | DF2K | HAT-S_SRx2 | 800k |
| [ART](https://github.com/gladzhang/ART) |   16.14    | 415.51 | DF2K | CAR_ART_q10 | 200k |
| [OAPT](https://arxiv.org/abs/2408.11480) |   12.96    | 293.60 | DF2K | 006_CAR_DFWB_s126w7_SwinIR-M_jpeg10 | 200k |







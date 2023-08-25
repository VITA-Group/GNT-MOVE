# Enhancing NeRF akin to Enhancing LLMs: Generalizable NeRF Transformer with Mixture-of-View-Experts [ArXiv](https://arxiv.org/abs/2308.11793)
[Wenyan Cong]()<sup>1∗</sup>,
[Hanxue Liang]()<sup>2,1∗</sup>,
[Peihao Wang](https://peihaowang.github.io/)<sup>1</sup>,
[Zhiwen Fan]()<sup>1</sup>,
[Tianlong Chen](https://tianlong-chen.github.io/)<sup>1</sup>,
[Mukund Varma T](https://mukundvarmat.github.io/)<sup>3,1</sup>,
[Yi Wang]()<sup>1</sup>,
[Zhangyang Wang](https://vita-group.github.io/)<sup>1</sup>

<sup>1</sup>University of Texas at Austin, <sup>2</sup>University of Cambridge, <sup>3</sup>Indian Institute of Technology Madras

<sup>*</sup> denotes equal contribution.

This repository is built based on GNT's [offical repository](https://github.com/VITA-Group/GNT)


## Introduction

Cross-scene generalizable NeRF models, which can di- rectly synthesize novel views of unseen scenes, have be- come a new spotlight of the NeRF field. Several existing attempts rely on increasingly end-to-end “neuralized” ar- chitectures, i.e., replacing scene representation and/or ren- dering modules with performant neural networks such as transformers, and turning novel view synthesis into a feed- forward inference pipeline. While those feedforward “neu- ralized” architectures still do not fit diverse scenes well out of the box, we propose to bridge them with the powerful Mixture-of-Experts (MoE) idea from large language models (LLMs), which has demonstrated superior generalization ability by balancing between larger overall model capacity and flexible per-instance specialization. Starting from a re- cent generalizable NeRF architecture called GNT, we first demonstrate that MoE can be neatly plugged in to en- hance the model. We further customize a shared permanent expert and a geometry-aware consistency loss to enforce cross-scene consistency and spatial smoothness respec- tively, which are essential for generalizable view synthesis. Our proposed model, dubbed GNT with Mixture-of-View-Experts (GNT-MOVE), has experimentally shown state-of- the-art results when transferring to unseen scenes, indicat- ing remarkably better cross-scene generalization in both zero-shot and few-shot settings.

## Installation

Clone this repository:

```bash
git clone https://github.com/VITA-Group/GNT-MOVE.git
cd GNT-MOVE/
```

The code is tested with python 3.8, cuda == 11.1, pytorch == 1.10.1. Additionally dependencies include: 

```bash
torchvision
ConfigArgParse
imageio
matplotlib
numpy
opencv_contrib_python
Pillow
scipy
imageio-ffmpeg
lpips
scikit-image
```

## Datasets

We reuse the training, evaluation datasets from [IBRNet](https://github.com/googleinterns/IBRNet). All datasets must be downloaded to a directory `data/` within the project folder and must follow the below organization. 
```bash
├──data/
    ├──ibrnet_collected_1/
    ├──ibrnet_collected_2/
    ├──real_iconic_noface/
    ├──spaces_dataset/
    ├──RealEstate10K-subset/
    ├──google_scanned_objects/
    ├──nerf_synthetic/
    ├──nerf_llff_data/
```
We refer to [IBRNet's](https://github.com/googleinterns/IBRNet) repository to download and prepare data. For ease, we consolidate the instructions below:
```bash
mkdir data
cd data/

# IBRNet captures
gdown https://drive.google.com/uc?id=1rkzl3ecL3H0Xxf5WTyc2Swv30RIyr1R_
unzip ibrnet_collected.zip

# LLFF
gdown https://drive.google.com/uc?id=1ThgjloNt58ZdnEuiCeRf9tATJ-HI0b01
unzip real_iconic_noface.zip

## [IMPORTANT] remove scenes that appear in the test set
cd real_iconic_noface/
rm -rf data2_fernvlsb data2_hugetrike data2_trexsanta data3_orchid data5_leafscene data5_lotr data5_redflower
cd ../

# Spaces dataset
git clone https://github.com/augmentedperception/spaces_dataset

# RealEstate 10k
## make sure to install ffmpeg - sudo apt-get install ffmpeg
git clone https://github.com/qianqianwang68/RealEstate10K_Downloader
cd RealEstate10K_Downloader
python3 generate_dataset.py train
cd ../

# Google Scanned Objects
gdown https://drive.google.com/uc?id=1w1Cs0yztH6kE3JIz7mdggvPGCwIKkVi2
unzip google_scanned_objects_renderings.zip

# Blender dataset
gdown https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
unzip nerf_synthetic.zip

# LLFF dataset (eval)
gdown https://drive.google.com/uc?id=16VnMcF1KJYxN9QId6TClMsZRahHNMW5g
unzip nerf_llff_data.zip
```

## Usage
<!-- 
### Training

```bash
# single scene
# python3 train.py --config <config> --train_scenes <scene> --eval_scenes <scene> --optional[other kwargs]. Example:
python3 train.py --config configs/gnt_blender.txt --train_scenes drums --eval_scenes drums
python3 train.py --config configs/gnt_llff.txt --train_scenes orchids --eval_scenes orchids

# cross scene
# python3 train.py --config <config> --optional[other kwargs]. Example:
python3 train.py --config configs/gnt_full.txt 
```

To decode coarse-fine outputs set `--N_importance > 0`, and with a separate fine network use `--single_net = False`

### Pre-trained Models

<table>
  <tr>
    <th>Dataset</th>
    <th>Scene</th>
    <th colspan=2>Download</th>
  </tr>
  <tr>
    <th rowspan=8>LLFF</th>
    <td>fern</td>
    <td><a href="https://drive.google.com/file/d/18wWmOh4v0yFP9Q3nyqpN82N-szYFJrf8/view?usp=sharing">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/12AcHS17HwVfFYMVX_t6dQU5c5jXOxtWg?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <td>flower</td>
    <td><a href="https://drive.google.com/file/d/1JPNHvCsQljUDPFZwrZ0KoxJWGgb1ik-H/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/1xbVFzEEcJtaFJaewdXvaScUpSDhKpom9?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <td>fortress</td>
    <td><a href="https://drive.google.com/file/d/1rDS3Ci0L4mhb2ju-2iqeLwC8fokzuM9I/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/1AIkIZw1drGjYyZaK8048FWGjDXhnNaKA?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <td>horns</td>
    <td><a href="https://drive.google.com/file/d/13hszXGhJ4Z9k3-NIJ9TlwSpw9c1zuzuW/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/1_CeRcd5VLFa1_NWIGu1qp2EN2GEmD3df?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <td>leaves</td>
    <td><a href="https://drive.google.com/file/d/1wi4WA39lU0pdhkbyXlFePX9Vz8nSsDpe/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/1UXMW9_8eellesWkhP_VcIcC6VQy1QCnB?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <td>orchids</td>
    <td><a href="https://drive.google.com/file/d/1RM0eZuF3Jn6Jpfd_LvixVcUaLNtpyKbX/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/1Wuxp1_mM8TQh5j8W1GHzGVmFZVbl0gul?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <td>room</td>
    <td><a href="https://drive.google.com/file/d/1DWtcPxMv7UceRkUrnRTKZ_-0RcxSnn12/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/1tlNBWH304jyBjbE8NCw1ysvtU53mglLg?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <td>trex</td>
    <td><a href="https://drive.google.com/file/d/1j2JQ7MkuWQe8vAaatFfRzFROLTZf9dba/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/1pW0Di9nE8q5KqffL7fVze2Wu_Jts8mAW?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <th rowspan=8>Synthetic</th>
    <td>chair</td>
    <td><a href="https://drive.google.com/file/d/1kSwVw03Df2JJbl-tkDgt03RcnZ8aXKPP/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/1pKeJmH4jMrnjbN3uELVlddfxSzoQDuCz?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <td>drums</td>
    <td><a href="https://drive.google.com/file/d/1YgUopHb5LXwmXlB7CDC7DF0bwjprH15W/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/10BAz_FmOFEEySKn__LqVcFVudNCRUie-?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <td>ficus</td>
    <td><a href="https://drive.google.com/file/d/1vizXtpTWmmPcZhWOzMXYXwM-7ReQbfuX/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/1uDr7ocb-9RlpK9L6vgxbC5d4g53H7WY1?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <td>hotdog</td>
    <td><a href="https://drive.google.com/file/d/1kjAi7Ff9lAnBZyWfmvH4APg-Kg508SaZ/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/1NHVZFSBIoVkNsrR7teSt7OVVJJVF9oaO?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <td>lego</td>
    <td><a href="https://drive.google.com/file/d/1IbhbBr5XfxQz0jSQM3nLX_htTbvc59kj/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/1BHzWiCNmGwa2FmgFAqql1SC7jkHM1clK?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <td>materials</td>
    <td><a href="https://drive.google.com/file/d/13H6SzaHCj6NbB0BgNkE8kVRjFOZys4dx/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/1rxCI5F-36gBUv6wO3REcGZs396YVm_7d?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <td>mic</td>
    <td><a href="https://drive.google.com/file/d/1fxHOPPKD1SaSy8aDC3iIDS41Rbkui1r9/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/1m64tU7Kl37Y6ToDFrJ65_OcMKbgpVpBq?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <td>ship</td>
    <td><a href="https://drive.google.com/file/d/16nLEu0pINfPJ46MbDkxgOEqnWo8hqAAF/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/1SQCCDxUdzlBJuagvRMkW0uowshNqY9xV?usp=share_link">renders</a></td>
  </tr>
  <tr>
    <td>generalization</td>
    <td>N.A.</td>
    <td><a href="https://drive.google.com/file/d/1AMN0diPeHvf2fw53IO5EE2Qp4os5SkoX/view?usp=share_link">ckpt</a></td>
    <td><a href="https://drive.google.com/drive/folders/1XW-uCao0WRyf5I94pdhW2H2wIcwZPrAu?usp=share_link">renders</a></td>
  </tr>
</table>

To reuse pretrained models, download the required checkpoints and place in appropriate directory with name - `gnt_<scene-name>` (single scene) or `gnt_<full>` (generalization). Then proceed to evaluation / rendering. To facilitate future research, we also provide half resolution renderings of our method on several benchmark scenes. Incase there are issues with any of the above checkpoints, please feel free to open an issue. 

### Evaluation

```bash
# single scene
# python3 eval.py --config <config> --eval_scenes <scene> --expname <out-dir> --run_val --optional[other kwargs]. Example:
python3 eval.py --config configs/gnt_llff.txt --eval_scenes orchids --expname gnt_orchids --chunk_size 500 --run_val --N_samples 192
python3 eval.py --config configs/gnt_blender.txt --eval_scenes drums --expname gnt_drums --chunk_size 500 --run_val --N_samples 192

# cross scene
# python3 eval.py --config <config> --expname <out-dir> --run_val --optional[other kwargs]. Example:
python3 eval.py --config configs/gnt_full.txt --expname gnt_full --chunk_size 500 --run_val --N_samples 192
```

### Rendering

To render videos of smooth camera paths for the real forward-facing scenes.

```bash
# python3 render.py --config <config> --eval_dataset llff_render --eval_scenes <scene> --expname <out-dir> --optional[other kwargs]. Example:
python3 render.py --config configs/gnt_llff.txt --eval_dataset llff_render --eval_scenes orchids --expname gnt_orchids --chunk_size 500 --N_samples 192
```

The code has been recently tidied up for release and could perhaps contain tiny bugs. Please feel free to open an issue. -->


## Cite this work

If you find our work / code implementation useful for your own research, please cite our paper.

```
@inproceedings{
    gntmove2023,
    title={Enhancing Ne{RF} akin to Enhancing {LLM}s: Generalizable Ne{RF} Transformer with Mixture-of-View-Experts},
    author={Wenyan Cong and Hanxue Liang and Peihao Wang and Zhiwen Fan and Tianlong Chen and Mukund Varma and Yi Wang and Zhangyang Wang},
    booktitle={ICCV},
    year={2023}
}
```

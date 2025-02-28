# [3DV 2025] A2-GNN: Angle-Annular GNN for Visual Descriptor-free Camera Relocalization

Authors: Yejun Zhang, Shuzhe Wang, Juho Kannala

##### Abstract

##### Pipeline

Environment Setup

```
git clone https://github.com/YejunZhang/a2-gnn.git
cd a2-gnn
conda env create -f environment.yml
conda activate a2-gnn
```

We need to install the corresponding ```torch_scatter=2.0.8```

```
wget https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_scatter-2.0.8-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.8-cp37-cp37m-linux_x86_64.whl
```

Now install A2-GNN

```
pip install . --find-links https://data.pyg.org/whl/torch-1.8.0+cu11.1.html
```

## Data Preparation

We use the same dataset as [DGC-GNN](https://github.com/AaltoVision/DGC-GNN-release) and the processed MegaDepth data can be found [here](https://drive.google.com/drive/folders/1ae8CHU42wTJleRrlG9GBY4V-PIdqsM0O?usp=sharing). Please download the dataset and put it to ```/data/MegaDepth_undistort```

## Training & Evaluation

```
# Train on MegaDepth
sh train.sh

# Eval on MegaDepth
# Specify model and output path
sh eval.sh
```

## Acknowledgements

We appreciate the previous open-source repository [GoMatch](https://github.com/dvl-tum/gomatch) and [CLNet](https://github.com/sailor-z/CLNet).

## Citation

Please consider citing our papers if you find this code useful for your research:

```
@misc{zhang2025a2gnnangleannulargnnvisual,
      title={A2-GNN: Angle-Annular GNN for Visual Descriptor-free Camera Relocalization}, 
      author={Yejun Zhang and Shuzhe Wang and Juho Kannala},
      year={2025},
      eprint={2502.20036},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.20036}, 
}
```

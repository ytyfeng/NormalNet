# NormalNet: Using PointNet for Normal Estimation in 3D Point Clouds

## Ty Feng, Yasaman Hajnorouzali, Mahdis Rabbani

Project Report: [Project_Report.pdf](Project_Report.pdf)

Presentation Video: [presentation.mp4](presentation.mp4)

To run:

```
cd pclouds
python download_pclouds.py
cd ..
python train_normalnet.py --indir ./pclouds --nepoch 100 --weight_decay 0.01 

```

To try different symmetric functions, use `--sym_op sum` or `--sym_op max`.


To use global features only as the NormalNet input, use `--global_feature True`. By default, both local and global features are used. 


The pretrained model is models/Normal_estimation_model_99.pth.


The train and test losses for our experiments are in losses_log folder. loss_plots.ipynb shows the loss plots. 


* This repo is based on this Pytorch implementation of PointNet, https://github.com/fxia22/pointnet.pytorch. 

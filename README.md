# ECS 271 project

## Ty Feng, Yasaman Hajnorouzali, Mahdis Rabbani

To run:

```
cd pclouds
python download_pclouds.py
cd ..
python train_normalnet.py --indir ./pclouds --nepoch 100 --weight_decay 0.01 --sym_op sum 

```

To evaluate:
```
python eval_normalnet.py --model /models/Normal_estimation_model_99.pth

```
# Multimodal-Final-Project
## introduction
this project is the final project of Peking University course multimodal learning
In this project we replicated the results in the paper _Text-Only Training for Image Captioning using Noise-Injected CLIP and improved it
the link of the code of origin paper is [here](https://github.com/DavidHuji/CapDec)
## Environment configuration
```
git clone https://github.com/DavidHuji/CapDec && cd CapDec #TODO replace with our git link
conda env create -f others/environment.yml
conda activate CapDec
```
## prepare data
Download the datasets using the following links: [COCO](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits)
edit the parse_karpathy.py with datasets path
```
python parse_karpathy.py
```
## traing
```
python embeddings_generator.py  --clip_model_type RN50  --dataset_mode 0
python train.py --data clip_embeddings_of_last_stage.pkl --out_dir ./coco_train/ --noise_variance 0.016
```
## evaluation

download the picture of coco dataset into data/coco
```
wget http://images.cocodataset.org/zips/val2014.zip
```
get the inference result
```
python predictions_runner.py  --checkpoint path_to_checkpoints.pt --dataset_mode 0
```
if you want to check many checkpoints at the same time you can copy the test.sh into the folder ,change the path in it and run the test.sh
get the metric score
```
cd coco-caption
python evaluation.py  --res ./results/res.json  --outpath data_res/res.txt
```
if you want to evaluate many results at the same time you can copy the evaluate.sh into the folder ,change the path in it and run the evaluate.sh

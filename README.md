# Multimodal-Final-Project


## Introduction

This is the final project of the course "Multimodal Learning" from Peking University.

In this project, we reproduce the results in the paper "Text-Only Training for Image Captioning using Noise-Injected CLIP" and enhance its performance.
The link of the code of origin paper is [here](https://github.com/DavidHuji/CapDec).

- `Code` is our implementation based on the original CapDec. Don't forget to `cd Code` before starting!
- `visualization` lists some graphs and results of our experiments, which are also contained in our report.
- `Multimodal Final Project Report.pdf` is our report written by Latex.
- `paper.pdf` is the original paper.
- `details.docx` has some personal analysis towards the original code of the paper.
- `pre.pptx` is our slices of the presentation.

## Environment configuration
```
git clone https://github.com/lhy101/Multimodal-Final-Project.git 
cd Multimodal-Final-Project/Code
conda env create -f others/environment.yml
conda activate CapDec
```

## Prepare Text-only Data

You can download the COCO dataset using the following link: [COCO](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits). Note that it only consits of the captions in COCO. You can parse the COCO dataset using parse_karpathy.py, which we have already done.

If you want to repeat the parsing process, don't forget to edit the dataset path in parse_karpathy.py to your own. 
```
python parse_karpathy.py
```

## Training

There are loads of training methods, which we have elaborated in our `Multimodal Final Project Report.pdf`. We highly recommend you to read our report firstly.

If you only want to try the text-only training methods, you don't need to download the images of the COCO training set. Otherwise, you need to download them and move them to `Code/data/coco/train2014`, from which we can extract these images.
```
wget http://images.cocodataset.org/zips/train2014.zip
```



```
python embeddings_generator.py  --clip_model_type RN50  --dataset_mode 0
python train.py --data clip_embeddings_of_last_stage.pkl --out_dir ./coco_train/ --noise_variance 0.016
```
## Evaluation

Download the picture of coco dataset into data/coco
```
wget http://images.cocodataset.org/zips/val2014.zip
```
Get the inference result
```
python predictions_runner.py  --checkpoint path_to_checkpoints.pt --dataset_mode 0
```
If you want to check many checkpoints at the same time you can copy the test.sh into the folder ,change the path in it and run the test.sh
get the metric score
```
cd coco-caption
python evaluation.py  --res ./results/res.json  --outpath data_res/res.txt
```
If you want to check many checkpoints at the same time you can copy the evaluate.sh into the folder ,change the path in it and run the evaluate.sh

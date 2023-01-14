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

## Environment Configuration
```
git clone https://github.com/lhy101/Multimodal-Final-Project.git 
cd Multimodal-Final-Project/Code
conda env create -f others/environment.yml
conda activate CapDec
```

## Prepare Text-only Data

You can download the COCO dataset using the following link: [COCO](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits). Note that it only consits of the captions in COCO. You can parse the COCO dataset using parse_karpathy.py, which we have already done. If you want to repeat the parsing process, don't forget to edit the dataset path in parse_karpathy.py to your own. 
```
python parse_karpathy.py
```

## Training

There are loads of training methods, which we have elaborated in our `Multimodal Final Project Report.pdf`. We highly recommend you to read our report first.

Since one of our methods (auxiliary training) need partial images from COCO training set, you need to download the images of the COCO training set and move them to `Code/data/coco/train2014`, from which we can extract these images.
```
wget http://images.cocodataset.org/zips/train2014.zip
```

Then, you can run `embeddings_generator.py` to generate the CLIP embeddings of the text as well as the images. It will take a long time even if you have a GPU. Thus, we highly recommend you to download the CLIP embeddings from [here](https://pan.baidu.com/s/1Fq40LnUS4Q-WW7WPdjyTFQ?pwd=0115), using password `0115`. You need to move the four `.pkl` documents to `Code/data/coco`, from which we can successfully extract them.

```
python embeddings_generator.py
```


### Training $N(0, 0.016)$
This is the reproduction of the original model, which has the best performance in the paper. We set it as a baseline.
```
python train.py --data COCO --out_dir ./coco_train/ --noise_variance 0.016
```
### Training $N(0, 0.016) w/o norm$
For every following models, you can add `--dont_norm` to train the model without using the normalizing trick before noise injection step. For instance, you can train the model below.
```
python train.py --data COCO --dont_norm --out_dir ./coco_train/ --noise_variance 0.016
```

### Training $U(0, 0.016)$
```
python train.py --data COCO --out_dir ./coco_train/ --noise_variance 0.016 --uniform_noise
```

### Training Models with Learnable Mean
```
python train.py --data COCO --out_dir ./coco_train/ --noise_variance 0.016 --uniform_noise
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

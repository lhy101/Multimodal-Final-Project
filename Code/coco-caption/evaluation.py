from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--res', default="./results/captions_val2014_fakecap_results.json")
parser.add_argument('--outpath', default="./data_res")
args = parser.parse_args()
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

import sys

if "/home/public/lhy/CapDec/coco-caption" not in sys.path:
    sys.path.append("/home/public/lhy/CapDec/coco-caption")
print(sys.path)

dataDir='.'
dataType='val2014'
algName = 'fakecap'
annFile='%s/annotations/captions_%s.json'%(dataDir,dataType)
subtypes=['results', 'evalImgs', 'eval']
[resFile, evalImgsFile, evalFile]= \
['%s/results/captions_%s_%s_%s.json'%(dataDir,dataType,algName,subtype) for subtype in subtypes]
resFile = args.res

# download Stanford models

# create coco object and cocoRes object
coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
cocoEval.evaluate()

# print output evaluation scores
with open(args.outpath,"w")as f:
    for metric, score in cocoEval.eval.items():
        f.write('%s: %.3f\n'%(metric, score)) 
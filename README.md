# image-generation-from-text-description-using-generative-adeverisal-network-GAN-
# DF-GAN plus attention : Deep Fusion Generative Adversarial Networks with spatial attention  for Text-to-Image Synthesis

(A novel and effective one-stage Text-to-Image Backbone)
 [ Deep Fusion Generative Adversarial Networks with spatial attention for Text-to-Image Synthesis]

---
### Requirements
- python 3.6+
- Pytorch 1.0+
- easydict
- nltk
- scikit-image
- A titan xp (set nf=32 in *.yaml) or a V100 32GB (set nf=64 in *.yaml)
### Installation


```

### Datasets Preparation
1. Download the preprocessed metadata for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`


### Pre-trained text encoder

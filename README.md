# Unsupervised Image Captioning
by [Yang Feng](http://cs.rochester.edu/u/yfeng23/), Lin Ma, Wei Liu, and
[Jiebo Luo](http://cs.rochester.edu/u/jluo)

### Introduction
Most image captioning models are trained using paired image-sentence data, which are 
expensive to collect. We propose unsupervised image captioning to relax the 
reliance on paired data. For more details, please refer to our
[paper](https://arxiv.org/abs/1811.10787).

![alt text](http://cs.rochester.edu/u/yfeng23/cvpr19_captioning/framework.png 
"Framework")

### Citation

    @InProceedings{feng2019unsupervised,
      author = {Feng, Yang and Ma, Lin and Liu, Wei and Luo, Jiebo},
      title = {Unsupervised Image Captioning},
      booktitle = {CVPR},
      year = {2019}
    }

### Requirements
```
pip install tensorflow-gpu
```

### Dataset
1. Crawl image descriptions. The descriptions used when conducting the 
experiments in the paper are available at
[link](https://drive.google.com/file/d/1z8JwNxER-ORWoAmVKBqM7MyPozk6St4M).
    ```
    python3 crawl_descriptions.py
    ```

2. The training code is coming soon.
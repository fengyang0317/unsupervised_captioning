# Unsupervised Image Captioning
by [Yang Feng](http://cs.rochester.edu/u/yfeng23/), Lin Ma, Wei Liu, and
[Jiebo Luo](http://cs.rochester.edu/u/jluo)

### Introduction
Most image captioning models are trained using paired image-sentence data, which
are expensive to collect. We propose unsupervised image captioning to relax the 
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
pip install -r requirements.txt
mkdir ~/workspace
cd ~/workspace
git clone https://github.com/tensorflow/models.git tf_models
git clone https://github.com/tylin/coco-caption.git
touch tf_models/research/im2txt/im2txt/__init__.py
touch tf_models/research/im2txt/im2txt/data/__init__.py
touch tf_models/research/im2txt/im2txt/inference_utils/__init__.py
wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
mkdir ckpt
tar zxvf inception_v4_2016_09_09.tar.gz -C ckpt
git clone https://github.com/fengyang0317/unsupervised_captioning.git
cd unsupervised_captioning
```

### Dataset (Optional. The files generated below can be found at [here][1]).
1. Crawl image descriptions. The descriptions used when conducting the
experiments in the paper are available at
[link](https://drive.google.com/file/d/1z8JwNxER-ORWoAmVKBqM7MyPozk6St4M).
You may download the descriptions from the link and extract the files to
data/coco.)
    ```
    pip3 install absl-py
    python3 preprocessing/crawl_descriptions.py
    ```

2. Extract the descriptions. It seems that NLTK is changing constantly. So 
the number of the descriptions obtained may be different.
    ```
    python -c "import nltk; nltk.download('punkt')"
    python preprocessing/extract_descriptions.py
    ```

3. Preprocess the descriptions.
    ```
    python preprocessing/process_descriptions.py --word_counts_output_file \ 
      data/word_counts.txt --new_dict
    ```

4. Download the MSCOCO images from [link](http://cocodataset.org/) and put 
all the images into ~/dataset/mscoco/all_images.

5. Object detection for the training images. You need to first download the
detection model from [here][detection_model] and then extract the model under
tf_models/research/object_detection.
    ```
    python preprocessing/detect_objects.py --image_path\
      ~/dataset/mscoco/all_images --num_proc 2 --num_gpus 1
    ```

6. Generate tfrecord files for images.
    ```
    python preprocessing/process_images.py --image_path\
      ~/dataset/mscoco/all_images
    ```

### Training
7. Train the model without the intialization pipeline.
    ```
    python im_caption_full.py --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt\
      --multi_gpu --batch_size 512 --save_checkpoint_steps 1000\
      --gen_lr 0.001 --dis_lr 0.001
    ```

8. Evaluate the model. The last element in the b34.json file is the best
checkpoint.
    ```
    CUDA_VISIBLE_DEVICES='0,1' python eval_all.py\
      --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt\
      --data_dir ~/dataset/mscoco/all_images
    js-beautify saving/b34.json
    ```

9. Evaluate the model on test set. Suppose the best validation checkpoint
is 20000.
    ```
    python test_model.py --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt\
      --data_dir ~/dataset/mscoco/all_images --job_dir saving/model.ckpt-20000
    ```

### Initialization (Optional. The files can be found at [here][1]).

10. Train a object-to-sentence model, which is used to generate the
pseudo-captions.
    ```
    python initialization/obj2sen.py
    ```

11. Find the best obj2sen model.
    ```
    python initialization/eval_obj2sen.py --threads 8
    ```

12. Generate pseudo-captions. Suppose the best validation checkpoint is 35000.
    ```
    python initialization/gen_obj2sen_caption.py --num_proc 8\
      --job_dir obj2sen/model.ckpt-35000
    ```

13. Train a captioning using pseudo-pairs.
    ```
    python initialization/im_caption.py --o2s_ckpt obj2sen/model.ckpt-35000\
      --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt
    ```

14. Evaluate the model.
    ```
    CUDA_VISIBLE_DEVICES='0,1' python eval_all.py\
      --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt\
      --data_dir ~/dataset/mscoco/all_images --job_dir saving_imcap
    js-beautify saving_imcap/b34.json
    ```

15. Train sentence auto-encoder, which is used to initialize sentence GAN.
    ```
    python initialization/sentence_ae.py
    ```

16. Train sentence GAN.
    ```
    python initialization/sentence_gan.py
    ```

17. Train the full model with initialization. Suppose the best imcap validation
checkpoint is 18000.
    ```
    python im_caption_full.py --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt\
      --imcap_ckpt saving_imcap/model.ckpt-18000\
      --sae_ckpt sen_gan/model.ckpt-30000 --multi_gpu --batch_size 512\
      --save_checkpoint_steps 1000 --gen_lr 0.001 --dis_lr 0.001
    ```

### Credits
Part of the code is from 
[im2txt](https://github.com/tensorflow/models/tree/master/research/im2txt),
[tfgan](https://github.com/tensorflow/models/tree/master/research/gan),
[resnet](https://github.com/tensorflow/models/tree/master/official/resnet),
[Tensorflow Object Detection API](
https://github.com/tensorflow/models/tree/master/research/object_detection) and
[maskgan](https://github.com/tensorflow/models/tree/master/research/maskgan).

[Xinpeng](https://github.com/chenxinpeng) told me the idea of self-critic, which
is crucial to training.

[1]: https://drive.google.com/drive/folders/1ol8gLj6hYgluldvdj9XFKm16TCqOr7EE
[detection_model]: http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28.tar.gz
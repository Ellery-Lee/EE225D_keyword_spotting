This is a repo for Chinese-keyword-spotting project.

### Dataset
We use the CAS-VSR-W1k(originally called LRW-1000) dataset.
https://drive.google.com/drive/u/1/folders/17TPjaHx2hOcPBGnl89s2G7DUDB6Wk67Q

### Feature extraction
- audio
  - main_audio.py: get features and returns a feature.pkl file. Feature extractor is audio_model.py, using librosa and Wav2Vec2 package.
  ```bash
  python main_audio.py     --gpus='0'      --lr=0.0     --batch_size=128     --num_workers=2     --max_epoch=120     --test=True     --save_prefix='checkpoints/lrw-1000-final/'     --n_class=1000     --dataset='lrw1000'     --border=True     --mixup=False     --label_smooth=False     --se=True     --weights='checkpoints/lrw1000-border-se-mixup-label-smooth-cosine-lr-wd-1e-4-acc-0.56023.pt'
  ```
- visual
  - prepare_lrw1000.py: data prepocessing, saves images in pkl files.
  - main_visual.py: get features and returns a feature.pkl file. Each data line in the file has a tuple with a format of ('filename', feature matrix in torch tensor). The pipeline of visual feature extractor is visual_extractor -> video_model -> main_visual
  ```bash
  python main_visual.py     --gpus='0'      --lr=0.0     --batch_size=128     --num_workers=2     --max_epoch=120     --test=True     --save_prefix='checkpoints/lrw-1000-final/'     --n_class=1000     --dataset='lrw1000'     --border=True     --mixup=False     --label_smooth=False     --se=True     --weights='checkpoints/lrw1000-border-se-mixup-label-smooth-cosine-lr-wd-1e-4-acc-0.56023.pt'
  ```
  checkpoints can be downloaded from https://github.com/Fengdalu/learn-an-effective-lip-reading-model-without-pains
- processFeatureFiles.py: read the feature.pkl file and save features in to corresponding directories. The hierachy of directories is WORDNAME/SPLIT/filename.

### Similarity Map and CNN matching
- model/model.py: the similarity map classification model. 

### Training
```bash
python train_LRW1000.py --config=./config/lrw1000/train.json 
```

### Testing
```bash
python test_LRW1000.py --config=./config/lrw1000/eval.json --resume=/home/dongwang/EE225D_keyword_spotting/data/saved/models/lrw1000-train/2021-12-04_19-27-15/checkpoint-epoch50.pth
```
(you need to change the configuration file for audio, visual, and audio-visual; p_size and g_size in train.json should include the number of unknown characters.)

### Testing results
visual-only WER: 31.53%
audio-only WER: 29.44%



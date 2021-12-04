This is a repo for Chinese-keyword-spotting project.

### Dataset
We use the CAS-VSR-W1k(originally called LRW-1000) dataset. 

### Feature extraction
- audio
  - process_wav.py: process the raw wav file, and generate training and validation data based on the info pkl file. Data is saved in pkl files.
  - audio_extractor.py: batch_input_queue(), and valid_queue() are the main feature extractor functions. They return mel-spectrogram as audio feature.
- visual
  - prepare_lrw1000.py: data prepocessing, saves images in pkl files.
  - main_visual.py: get features and returns a feature.pkl file. Each data line in the file has a tuple with a format of ('filename', feature matrix in torch tensor). The pipeline of visual feature extractor is visual_extractor -> video_model -> main_visual
  - processFeatureFiles.py: read the feature.pkl file and save features in to corresponding directories. The hierachy of directories is WORDNAME/SPLIT/filename.

### Encoder

### Similarity Map and CNN matching
- kws.py: the similarity map classification model
- audio_similarity_map.py / visual_similarity_map.py: get the audio / visual features, and implement the kws.py model

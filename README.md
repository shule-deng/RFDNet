# RFDNet

## Citation
If you find Simple Shot useful in your research, please consider citing:





## Usage
### 1.Dependencies
(1) Python 3.8.10 <br/>
(2) Pytorch 1.8.1 <br/>
(3) CUDA 11.2 <br/>

### 2. Download Datasets
### 2.1 Mini-ImageNet
You can download the dataset from https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view

### 2.2 Tiered-ImageNet
You can download the dataset from https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view.

### 2.3 CIFAR-FS
You can download the dataset from https://drive.google.com/drive/folders/12RSelk-cW_nkOPkeux1TIcX53BBOfFyN?usp=sharing


### 3. Training
### 3.1 Pre-training
python pretrain.py <br/>
<br/>
you can also download our pre-trained models from https://drive.google.com/drive/folders/16tQ8XnGO2OhpD2DPpchoJ916-Vs01gbn

### 3.2 Meta-training
python train.py -c /path/to/config/file/dataset_name.config

## Contact
If you have any question, please feel free to email us. (audsl@mail.scut.edu.cn)








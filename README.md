# AudioRWKV: Pretrained Audio Mamba for Audio Pattern Recognition
## Introduction

The Code Repository for  "AudioMamba: Pretrained Audio RWKV for Audio Pattern Recognition"

## Getting Started

### Environments
The codebase is developed with pytorch == 1.8.1, torch-lightning == 1.5.9
Install requirements as follows:
```
pip install -r requirements.txt
```


### Download and Processing Datasets

* config.py
```
change the varible "dataset_path" to your audioset address
change the variable "desed_folder" to your DESED address
change the classes_num to 527
```

* [AudioSet](https://research.google.com/audioset/download.html)
```
./create_index.sh # 
// remember to change the pathes in the script
// more information about this script is in https://github.com/qiuqiangkong/audioset_tagging_cnn

python main.py save_idc 
// count the number of samples in each class and save the npy files
```
* [ESC-50](https://github.com/karolpiczak/ESC-50)
```
Open the jupyter notebook at esc-50/prep_esc50.ipynb and process it
```
* [Speech Command V2](https://arxiv.org/pdf/1804.03209.pdf)
```
Open the jupyter notebook at scv2/prep_scv2.ipynb and process it
```
* [DESED Dataset](https://project.inria.fr/desed/) 
```
python conver_desed.py 
// will produce the npy data files
```

### Set the Configuration File: config.py

The script *config.py* contains all configurations you need to assign to run your code. 
Please read the introduction comments in the file and change your settings.


### Training 
First config the model parameters in config.py

For tiny model
```
img_size = (992,64)
patch_size = 16
rwkv_embed_dim = 192
rwkv_num_heads = 3
rwkv_dropout_rates = 0.0
```

for small model
```
img_size = (992,64)
patch_size = 16
rwkv_embed_dim = 384
rwkv_num_heads = 6
rwkv_dropout_rates = 0.3
rwkv_init_values = 1e-5
```

For base model
```
img_size = (992,64)
patch_size = 16
rwkv_embed_dim = 768
rwkv_num_heads = 12
rwkv_dropout_rates = 0.5
rwkv_post_norm = True
rwkv_init_values = 1e-5
```

Start Training:
```
python main.py train
```

### Results
AudioSet: 
|     | RWKV tiny | RWKV small | RWKV base |
|-----|------|-------|------|
| mAP | 40.1  |      |      |
|      |       |      |      |
|      |       |      |      |


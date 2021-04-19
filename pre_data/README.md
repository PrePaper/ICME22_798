# Prepare data
The directory weibo_dataset contains the representations of data in Weibo dataset.
The gen_by_VGG.py file is for generating visual unimodal representations of images in Weibo and Twitter datasets. 
The gen_emb_Bert.py file is used to generate textual unimoda representations of text content in Weibo and Twitter datasets and prepare the representation files for MUGS.

# Note
The environment for keras-bert is different from the MUGS.
The required version of packages are in requirements.txt.
You can use the following commend in Anaconda to create a new environment.
```
$ conda create --name <env_name> --file requirements.txt
```

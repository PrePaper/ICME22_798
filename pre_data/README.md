# Prepare data
The directory weibo_dataset contains the representations of data in Weibo dataset.
The gen_emb_Bert.py and gen_by_VGG.py files are used to generate textual and visual unimodal representations from meta-data in Weibo and Twitter datasets. 

# Note
The environment for keras-bert is different from the MUGS.
The required version of packages are in requirements.txt.
You can use the following commend in Anaconda to create a new environment.
```
$ conda create --name <env_name> --file requirements.txt
```

# MM21-paper-id-2418
Pytorch code for paper 2418: "MUGS: Multimodal Rumor Detection by Multigranular Structure Learning"

# overview
This directory contains code necessary to run the MUGS. MUGS is a multimodal rumor detection network by multigranular structure learning. See our paper for details on the code.

# dataset
The meta-data of the Weibo and Twitter datasets used in our experiments are available in the reference papers. In this project, we provide the representations generated by the pretrained [keras-bert](https://github.com/CyberZHG/keras-bert) model and [VGG-19](https://chsasank.github.io/vision/models.html) model in the pre_data subdirectory for easy use.

# requirements
The detailed version of all packages is available in requirements.txt.

To guarantee that you have the right package versions, you can use the following command in Anaconda to create a new environment.
```
$ conda create --name <env_name> --file requirements.txt
```

# running the code
The train.py is the main file for running the code.
```
$ python train.py
```

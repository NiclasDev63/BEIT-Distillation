# Beit Distillation

## Description
Our goal was to create a model that is as small as possible while maintaining good accuracy on the [VizWizVQA challenge](https://vizwiz.org/tasks-and-datasets/vqa/), so that it could be used to run on personal devices with short inference time and help visually impaired people in their daily lives.
Towards this end, we finetuned the [BEIT-3](https://github.com/microsoft/unilm/blob/master/beit3/README.md) base and large model on the [VizWizVQA challenge](https://vizwiz.org/tasks-and-datasets/vqa/), followed by finetuning the base model using Knowledge Distillation with the large model as the teacher. 
The code used for training and evaluation can be found in this repository.

## Installation
```
git pull https://git.rwth-aachen.de/mai-beit/beit-distillation.git

git install .
```

### Setup VizWiz
In the repository folder create a folder called `VizWiz`. 
In this folder create four subfolders: `VizWizAnnotations`, `VizWizTestImages`, `VizWizTrainImages`, `VizWizValImages` where the respective parts of the VizWizVQA Dataset need to be downloaded to.

### Setup BEiT-3

Any base-model checkpoint should go in a `/models/base` folder. While checkpoints should follow our naming pattern, the base file from BEiT-3 however works as-is (for base and large).\
Any large-model checkpoint should go in a `/models/large` folder.\
Any KD-model should go into `/notebooks/kd_student`.

Make sure all of these folders exist before running the code to prevent errors. 

Our final checkpoints for the BEIT-3 Model can be found [here](https://drive.google.com/drive/folders/1yoo0cGRoOG1dWltSTIC9skkkjmlESGVH):
+ [BEIT-3 Base finetuned on VizWiz](https://drive.google.com/file/d/1cLzQFQOjPkOBgYPsaW85uJH6G4UkV6VY/view?usp=drive_link)
+ [BEIT-3 Large finetuned on VizWiz](https://drive.google.com/file/d/1CSI3ARjR-Fv400dsH0WCyGCMwm572UXY/view?usp=drive_link)
+ [BEIT-3 Base finetuned on VizWiz with finetuned BEIT-3 Large as teacher](https://drive.google.com/file/d/14nDadqfV_vitST04_yC-k4lqwSf_Gl_e/view?usp=drive_link)

## Usage
We recommend using the jupyter [notebooks](https://git.rwth-aachen.de/mai-beit/beit-distillation/-/tree/main/notebooks?ref_type=heads) for a quick overview on results (Beit3_testing is deprecated).

Alternatively, the file [Beit3_vizwiz.py](https://git.rwth-aachen.de/mai-beit/beit-distillation/-/blob/main/Beit3_vizwiz.py?ref_type=heads) in the main folder can also be used for evaluating a given model checkpoint, the checkpoint name and hyperparameters have to be adapted in the code though (similarly for finetuning with [beit3_vizwiz_finetuning.py](https://git.rwth-aachen.de/mai-beit/beit-distillation/-/blob/main/beit3_vizwiz_finetuning.py?ref_type=heads)).

## Authors and acknowledgment
Niclas Gregor \
Nick Hammerbacher \
Jonathan Lieske\
Leonard Nintz
# RTIC: Residual Learning for Text and Image Composition using Graph Convolutional Network
This is the official code of RTIC: Residual Learning for Text and Image Composition using Graph Convolutional Network. The code only supports training and evaluation on FashionIQ. We release the implementations for the other baselines together.

![banner](img/banner.png)

## Updates
+ (2021.09.10) The official code is released.

## Requirements
Prepare your environment with virtualenv.
~~~
python3 -m virtualenv --python=python3 venv # create virtualenv.
. venv/bin/activate # activate environment.
pip3 install -r requirements.txt # install require packages.
~~~

## Download Data
We provide script for downloading FashionIQ images.
Note that it does not ensure that all images can be downloaded because we found some urls are broken.

~~~
sh script/download_fiq.sh
~~~

## Model Zoo
We provide pretrained checkpoint of RTIC / RTIC-GCN trained on FashionIQ.
The checkpoints will be available soon.
Stay tuned!

Model | Recall | Checkpoint | Config | Training Log
-- | -- | -- | -- | --
RTIC | 39.22 | n/a | n/a | n/a
RTIC-GCN (scratch) | 39.55 | n/a | n/a | n/a
RTIC-GCN (finetune) | 40.64 | n/a | n/a | n/a

## Quick Start
We provide sample training script to run on different configurations.
The default configurations are stored in `cfg/default.yaml` which represents "unified environmet" in our paper.
To try with "optimal environment", please use `+optimize=<someting>` option.

**RTIC (unified env)**

~~~
EXPR_NAME=testrun python main.py \
    config.EXPR_NAME=${EXPR_NAME}
~~~

**RTIC (optimal env)**

~~~
EXPR_NAME=testrun python main.py \
    +optimize=rtic \
    config.EXPR_NAME=${EXPR_NAME}
~~~

**RTIC-GCN (optimal env, scratch)**

~~~
EXPR_NAME=testrun_gcn LOAD_FROM=testrun python main.py \
    +optimize=rtic_gcn_scratch \
    +gcn=enabled \
    config.LOAD_FROM=${LOAD_FROM} \
    config.EXPR_NAME=${EXPR_NAME}
~~~

**RTIC-GCN (optimal env, finetune)**

~~~
EXPR_NAME=testrun_gcn LOAD_FROM=testrun python main.py \
    +optimize=rtic_gcn_finetune \
    +gcn=enabled \
    config.LOAD_FROM=${LOAD_FROM} \
    config.EXPR_NAME=${EXPR_NAME}
~~~

**Any Other Baselines**

you can train any other baselines by simply changing `config.TRAIN.MODEL.composer_model.name`.

~~~
(w/o GCN)
EXPR_NAME=testrun python main.py \
    config.TRAIN.MODEL.composer_model.name=<any-composer-method-you-want-to-try> \
    config.EXPR_NAME=${EXPR_NAME}
~~~

~~~
(w GCN)
EXPR_NAME=testrun_gcn LOAD_FROM=testrun python main.py \
    +gcn=enabled \
    config.TRAIN.MODEL.composer_model.name=<any-composer-method-you-want-to-try> \
    config.LOAD_FROM=${LOAD_FROM} \
    config.EXPR_NAME=${EXPR_NAME}
~~~

## License

MIT

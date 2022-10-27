## Source code of paper "Transformer Uncertainty Estimation with Hierarchical Stochastic Attention" (AAAI 2022)

## 1. Introduction
This repository releases the source code of stochastic models proposed in paper "Transformer Uncertainty Estimation with Hierarchical Stochastic Attention", which is accepted by AAAI conference in 2022.
We implemented stochastic transformer models for the following 2 NLP tasks:
1. Sentiment Analysis (code/IMDB);
2. Linguistic Acceptability (code/CoLA);

## 2. Environment
We implemented the model based on pytorch 1.8.1 and python 3.7.6. And config experimental enviroment by the following steps.
```bash
conda create -n pytorch_latest_p37 python=3.7 anaconda  # creat the virtual environment
source activate pytorch_latest_p37                      # activate the environment
sh setup.sh                                             # install all dependent packages
```

## 3. Datasets
We have experiment on two datasets:
1. IMDB: https://pytorch.org/text/_modules/torchtext/datasets/imdb.html
2. COLA: https://nyu-mll.github.io/CoLA/

## 4. Quickstart Training
### 4.1 Sentiment Analysis
1. Preprocessing
```bash
python code/IMDB/Run.py --mode=pre --model_name=IMDB --model_type=tf-sto --exp_name=default --job_id=123456 --debug=0
```

2. Train & test N_RUN times with uncertainty
```bash
python code/IMDB/Run.py --mode=uncertain-train-test --model_name=IMDB --model_type=tf-sto --exp_name=single_t1  --debug=0
```

More details can be found at [code/IMDB/README.md](https://github.com/amzn/sto-transformer/blob/main/code/IMDB/README.md).

### 4.2 Linguistic Acceptability
1. Downloaded the CoLA dataset from the repository (https://github.com/pranavajitnair/CoLA)
2. Train and validate the model run:
```bash
python train.py --model_type sto_transformer  --inference True  --sto_transformer True --model_name dual --dual True
```
More details can be found at [code/CoLA/README.md](https://github.com/amzn/sto-transformer/blob/main/code/CoLA/README.md).

## Reference
- Emails: 
  - Jiahuan Pei, <jpei@amazon.com>
  - Cheng Wang, <cwngam@amazon.com>
  - György Szarvas, <szarvasg@amazon.com>
- Paper
  - [Direct link](https://assets.amazon.science/1a/48/cb3245fb448ba775f163f02c2e6b/transformer-uncertainty-estimation-with-hierarchical-stochastic-attention.pdf)
  - Citation with bibtex
```bibtex
@inproceedings{pei2022transformer,
    title={Transformer uncertainty estimation with hierarchical stochastic attention},
    author={Pei, Jiahuan and Wang, Cheng and Szarvas, Gy{\"o}rgy},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={36},
    number={10},
    pages={11147--11155},
    year={2022}
}
```
## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.


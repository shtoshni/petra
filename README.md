# PeTra
Code and data for the ACL 2020 paper [PeTra: A Sparsely-Supervised Memory Model for People Tracking](https://www.aclweb.org/anthology/2020.acl-main.481.pdf)

## Requirements
Install python3 requirements:`pip install -r requirements`

## Data
The gap files have been downloaded from the [GAP repo](https://github.com/google-research-datasets/gap-coreference)<br/>
We created the diagnostic test of counting unique people in a document for which we annotated 100 GAP validation instances. The annotation file is [data/num_people.tsv](https://github.com/shtoshni92/petra/blob/master/data/num_people.tsv)

## Training/Validation steps
```
git clone https://github.com/shtoshni92/petra.git
cd petra/
export PYTHONPATH=${PWD}/src:$PYTHONPATH
python src/experiments/main.py -model_size [base/large] -mem_type [vanilla/learned/key_val] -num_cells [10/20] -data_dir data/ -base_model_dir DIRECTORY_TO_STORE_ALL_MODELS
```
Since we don't finetune BERT, all experiments can be done on a 12GB GPU. <br/>
Evaluation will be automatically done at the end of the training. Passing the `-eval ` flag is another way to perform evaluation.

## Important Hyperparams
* model_size: Specify the size of the BERT model between base and large.
* mem_type: Type of memory cell architecture: 'vanilla' -> PeTra, 'learned' -> PeTra + Learned Init., 'key_val' -> PeTra + Fixed Key
* num_cells: Vary the memory size

## Inference - Colab
[Here's a Colab notebook](https://colab.research.google.com/drive/17xT1QKCbj_tOFpiszHxuLkhjXLPp_hkd?usp=sharing) where we perform inference with a pretrained model and visualize the memory logs. 

## Citation
```
@inproceedings{toshniwal2020petra,
    title = {{PeTra: A Sparsely Supervised Memory Model for People Tracking}},
    author = "Shubham Toshniwal and Allyson Ettinger and Kevin Gimpel and Karen Livescu",
    booktitle = "ACL",
    year = "2020",
}
```

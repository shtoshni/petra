# PeTra
Code and data for the ACL 2020 paper [PeTra: A Sparsely-Supervised Memory Model for People Tracking](https://www.aclweb.org/anthology/2020.acl-main.481.pdf)

## Requirements
```
pip install -r requirements
```

## Setup steps
```
git clone https://github.com/shtoshni92/petra.git
export PYTHONPATH=petra/src:$PYTHONPATH
cd petra/src/experiments
python main.py -model_size [base/large] -mem_type [vanilla/learned/key_val] -num_cells [10/20] -data_dir ../../data -base_model_dir DIRECTORY_TO_STORE_ALL_MODELS
```



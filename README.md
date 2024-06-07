# MD3MD
## Dependencies
```
# Clone the environment
conda env create -f MD3MD.yml

# Activate the environment
conda activate MD3MD
```
## Data preparation
### QM9 dataset preparation
```
cd ./qm9/data/prepare/

python ./qm9/data/prepare/download.py
```
### Geom dataset preparation
1. Download the file at https://dataverse.harvard.edu/file.xhtml?fileId=4360331&version=2.0: wget https://dataverse.harvard.edu/api/access/datafile/4360331

2. Untar it and move it to data/geom/ tar -xzvf 4360331

3. pip install msgpack

4. python3 build_geom_dataset.py --conformations 1
## Unconditional training and sampling
```
# Training on QM9
python train.py --config './configs/qm9_full_epoch.yml'

# Training on Geom
python train.py --config './configs/geom_full.yml'

# Sampling and evaluation
python test_eval.py --ckpt <path> --sampling_type generalized --w_global_pos 1 -- w_global_node 1 --w_local_pos 4 --w_local_node 5
```
## Conditional training and sampling
### Train a specific classifier
```
cd qm9/property_prediction

python main_qm9_prop.py --num_workers 2 --lr 5e-4 --property alpha --exp_name exp_class_alpha --model_name egnn
```
### Train a conditional model
```
python train_qm9_condition.py --config './configs/qm9_full_epoch.yml' --context {property name-[homo | lumo | alpha | gap | mu | Cv]} --config_name {config_name}
```
### Sampling for MAE evaluation
```
python eval_qm9_condition_update.py --ckpt {saved_chekpoint} --classifiers_path {saved_class_checkpoint}
```

# MANP: Multi-Modal Aided Negative Prototype for Few-Shot Open-Set Recognition
## 0.dataset and env

see https://github.com/shiyuanh/TANE

## 1.Pretrain
For MiniImageNet:
We provide the pre-trained models for MiniImageNet. Save the pre-trained model to <pretrained_model_path>. or you can train your pretrain model by:

**Phase1:**

backbone pre-training

```python
python ./pretrain/pretrain_model/batch_process.py --gpus 0 --dataset <dataset> --model_path <log_root>  --data_root <data_dir> --batch_size 128 
```
**Phase2:**

The pre-trained model is fixed, and the reverse classification task is trained to obtain open weights.

## 2.Meta-learn

## 3.test

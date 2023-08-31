# Correspondence Transformers With Asymmetric Feature Learning and Matching Flow Super-Resolution（ACTR）

This is the official code for ACTR implemented with PyTorch.

# Environment Settings
```
git clone https://github.com/YXSUNMADMAX/ACTR
cd ACTR
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U scikit-image
pip install git+https://github.com/albumentations-team/albumentations
pip install tensorboardX termcolor timm tqdm requests pandas
```

# Evaluation
- Download pre-trained weights on [Link](https://drive.google.com/drive/folders/1ooKn4hJ65N352wYuWOnMXnjaVT1e0pVn?usp=sharing)

Result on SPair-71k:
      python test.py --datapath "/path_to_dataset" --pretrained "/path_to_pretrained_model/spair" --benchmark spair

Results on PF-PASCAL:
      python test.py --datapath "/path_to_dataset" --pretrained "/path_to_pretrained_model/pfpascal" --benchmark pfpascal

# Acknowledgement <a name="Acknowledgement"></a>
We borrow code from public projects (huge thanks to all the projects). We mainly borrow code from  [CATs](https://github.com/SunghwanHong/Cost-Aggregation-transformers). 

### BibTeX
If you find this research useful, please consider citing:
````BibTeX
@inproceedings{sun2023correspondence,
  title={Correspondence Transformers With Asymmetric Feature Learning and Matching Flow Super-Resolution},
  author={Sun, Yixuan and Zhao, Dongyang and Yin, Zhangyue and Huang, Yiwen and Gui, Tao and Zhang, Wenqiang and Ge, Weifeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17787--17796},
  year={2023}
}
````

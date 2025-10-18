### Requirements
> - Python 3.8, PyTorch >= 1.11
> - Platforms: Ubuntu 18.04, cuda-11

### Installation
```
# Create the DFAN virtual environment
conda create -n DFAN python=3.8
conda activate DFAN
# Install PyTorch
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

### Training
Run the following commands for training:
```
# train BDE
python train.py
```
### Testing
- Download the pre-trained BDE model.
- Download the testing dataset.
- Run the following commands:
```
# test BDE
python test.py
```
### Results
- The test results will be in './results'.

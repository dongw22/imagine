### Create Environment
1. Create Conda Environment
```
conda create --name imagine python=3.10
conda activate imagine
```

2. Install Dependencies
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm
```


### Pre-trained Model
- [Pre-trained Model for NTIRE 2025 Low Light Image Enhancement Challenge](https://mcmasteru365-my.sharepoint.com/:u:/g/personal/dongw22_mcmaster_ca/EQOvdzV160JMv-yge2bM3TcBG9Z6oOb4HyTlB4L4aDgsOw?e=8JfVC0).

### Our Submission on Test Sever
- [Our Test Output](https://mcmasteru365-my.sharepoint.com/:f:/g/personal/dongw22_mcmaster_ca/EpN54Q4bzO9DteK4tntg_eYB4X8XzlqI8A6HNXtAUEALSw?e=7zbUrM).

### Testing
Download above saved models and put it into the folder ./Enhancement/weights. To test the model, you need to specify the input image path (`args.input_dir`), the input structure prior path(`args.input_dir_s`), and pre-trained model path (`args.weights`) in `./Enhancement/test.py`. Then run
```bash
cd Enhancement
python test.py 
```
You can check the output in `test-results-ntire25`.


### Contact
If you have any question, please feel free to contact us via dongw22@mcmaster.ca.


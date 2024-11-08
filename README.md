# 1. Data
```
data    __train __xedap_0000.jpg
                __.....   
        __test 
```
- download data
```
chua cap nhat
```
# 2. Train
```
Cài đặt các thư viện như pytorch, numpy, opencv-python, tdqm, torchvision ...
Khuyến khích sử dụng anaconda
```

```
python train.py --data_root datasets/train
```

# 3. Test

```
python predict.py --checkpoint_dir checkpoints --image data/test/image.jpg
``` 
# motorcycles_bicycles_classifier
# motocycles_bicycles_classifier
# motorbikes_bikes_classifier

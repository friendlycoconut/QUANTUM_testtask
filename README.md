# QUANTUM_testtask


Dataset source [here](https://www.kaggle.com/c/data-science-bowl-2018/data)

# Process of model training
Metrics
```
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
```

# Running commands
```
git clone https://github.com/friendlycoconut/QUANTUM_testtask.git

```
Py 3.8.2
```
For training model startup:
```
python train.py
```

For cheching results of model
```
python predict_masks.py
```

# License

Copyright (c) Illia Kostenko
Licensed under the [MIT](https://github.com/friendlycoconut/QUANTUM_testtask/blob/master/LICENCE) License.

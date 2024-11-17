Initial Model 
- over fitting 
- validation accuracy = 10% 
- 3 convolutional blcoks 
- no regularization 
- no augmentation layers 
- pre procesing V1 
- regular relu layer changed to leaky 13% -> 16% test accuracy
- training reaches 100% accuracy very quickly in the beginning because of the over fitting 
- 100 epochs 32 batches -> 60% train accuracy 
Model v2 
- relu layer 
- augmentation layer added (randomization)
- 3 convolutional blocks followed by dense layers 
- each block is now more intensive 
Model v3 
- each of us will make a different model file and attempt to adjust the layers, etc to 
try and increase the accuracy 



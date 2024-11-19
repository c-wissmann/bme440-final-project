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
    - Model v3 - H
        ** added a max pooling layer as that seems something commonly used in image NNs 
        ** got a 0.5882 accuracy 
        ** made the max pooling (3,3) dropped accuracy
        ** adding another maxpooling (2,2) and changing the other back dropped accuracy as well
        ** ran with maxpooling again and it has a dropped accuracy again??? 
        ** ran again without changing any code and got the 0.5882 accuracy again???
        ** chris already had maxpooling layers, im an idiot 
        ** sum pooling hte hte middle converges to fast, going to attempt to replace exisitng
            max pooling with sum pooling 
        ** replacing max pooling with averagepooling has a better loss validation curve,
            but the same exact accuracy as previous tests 
- the model seems to know scoliosis, as the percentage of our data that is accuracte is 
    the exact percent of scolioisis, attempt to remove spondolythicus (idfc how to spell it) 
    and test 
- running chris's after mine seems to give the 027 accuracy then running it one more time 
    fizes the issue 
- attempt to change data sets, IE remove spondo, or classify spondo or scoliosis as 
    both unhealthy and then normal as normal to attempt to shake up the model 






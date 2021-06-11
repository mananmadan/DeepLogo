## Custom Testing Process
- test-set.py file
- this file reads all the images from the given input dir and then verifies the result using the annotations

## Performance of the Model
- Training Images: 3624
- Testing Images: 449
- Accuracy of the model on the test set: 98.2%
- TP: 441
- FP(Some value detected .. but not the correct one): 7
- TN(No value detected and no logo) = 0  
- FN: 0 (No value detected and logo) = (all cases where no value detected)
## Custom Testing Process
- test-set.py file
- this file reads all the images from the given input dir and then verifies the result using the annotations

## Performance of the Model
- Total training images: 781
- Total Testing images: 27
- Accuracy of the model on the test set: 98.2%

## Model Used
- DeepLogo uses SSD as a backbone network and fine-tunes pre-trained SSD released in the tensorflow/models repository.
- [SSD](https://arxiv.org/abs/1512.02325) 

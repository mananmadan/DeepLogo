import numpy as np
import matplotlib.pyplot as plt

## training and text annotation files
train_file = 'flickr_logos_27_dataset_training_set_annotation_cropped.txt'
test_file  = 'flickr_logos_27_dataset_test_set_annotation_cropped.txt'

## Load data from the annotation file
trdata = np.loadtxt(train_file, delimiter=',', dtype=str)
trclasses = trdata[:, -1] # last col has data regarding the classes

tedata = np.loadtxt(test_file,delimiter=',',dtype=str)
teclasses = tedata[:, -1]


## Load the list mapping index to name
cat_index = ["Adidas","Apple","BMW","Citroen","Cocacola","DHL","Fedex","Ferrari","Ford","Google","HP","Heineken","Intel","McDonalds","Mini","Nbc","Nike","Pepsi","Porsche","Puma","RedBull","Sprite","Starbucks","Texaco","Unicef","Vodafone","Yahoo"]


## Count the distribution
trdict = {}
tedict = {}
for i in trclasses:
   if int(i) not in trdict:
	   trdict[int(i)] = 1
   else:
	   trdict[int(i)] = trdict[int(i)]+1

for i in teclasses:
   if int(i) not in tedict:
	   tedict[int(i)] = 1
   else:
	   tedict[int(i)] = tedict[int(i)]+1

##Plot the graph
plt.bar(*zip(*tedict.items()))
plt.show()
plt.clf()
plt.bar(*zip(*trdict.items()))
plt.show()
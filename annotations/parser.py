import numpy as np
import matplotlib.pyplot as plt

## training and text annotation files
train_file = 'flickr_logos_27_dataset_training_set_annotation_cropped.txt'
test_file  = 'flickr_logos_27_dataset_test_set_annotation_cropped.txt'

## Load data from the annotation file
trdata = np.loadtxt(train_file, delimiter=',', dtype=str)
trclasses = trdata[:, -1] # last col has data regarding the classes
imgtr = trdata[:,0]
tedata = np.loadtxt(test_file,delimiter=',',dtype=str)
teclasses = tedata[:, -1]
imgte = tedata[:,0]


## Load the list mapping index to name
cat_index = ["Adidas","Apple","BMW","Citroen","Cocacola","DHL","Fedex","Ferrari","Ford","Google","HP","Heineken","Intel","McDonalds","Mini","Nbc","Nike","Pepsi","Porsche","Puma","RedBull","Sprite","Starbucks","Texaco","Unicef","Vodafone","Yahoo"]
cnt = 0
## avoid recounting
vis = {}

## Count the distribution
trdict = {}
tedict = {}
for i in range(0,len(trclasses)):
   cindex = int(trclasses[i])
   name = imgtr[i]
   if name in vis:
	   continue
   vis[name] = 1
   if int(cindex) not in trdict:
	   trdict[cindex] = 1
   else:
	   trdict[cindex] = trdict[cindex]+1

for i in range(0,len(teclasses)):
   cindex = int(teclasses[i])
   name = imgte[i]
   if name in vis:
	   continue
   vis[name] = 1
   if cindex not in tedict:
	   tedict[cindex] = 1
   else:
	   tedict[cindex] = tedict[cindex]+1

## Print freq by name
sum = 0
for i in trdict:
	sum = sum + trdict[i]
	print(cat_index[i],":",trdict[i])
print("Total training images:",sum)
sum = 0
for i in tedict:
	sum  = sum + tedict[i]
print("Total Testing images:",sum)
##Plot the graph
plt.bar(*zip(*tedict.items()))
plt.show()
plt.clf()
plt.bar(*zip(*trdict.items()))
plt.show()
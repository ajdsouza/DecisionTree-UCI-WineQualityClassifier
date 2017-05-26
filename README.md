Implementation of Decision Tree - Classify Wine Quality - UCI Dataset
---------------------------------------------------------------------
Implement the decision tree classifier using Python for classification of wine quality using Wine Quality dataset from UCI. Best performing tree has Danamic KTile Binning, Tree Pruning based on Cost Complexity, Entropy Gain for
selecting attribute for splitting


Problem
-------
Evaluate Performance of the classifier by 10-fold cross validation on a provided dataset. 

Use the wine quality dataset from the UCI machine learning repository available at
http://archive.ics.uci.edu/ml/datasets/Wine+Quality


The classification task is to determine whether a wine quality is over 7. 
The wine quality scores are mapped to binary classes of 0 and 1.
The Wine scores from 0 to 6 (inclusive) are mapped to 0, wine scores of 7 and above are mapped to 1. 


The data is stored in a comma separated file (csv). 
Each line describes a wine, using 12 columns: 
The first 11 describe the wine’s characteristics ( details ) 
The last column is a ground truth label for the quality of the wine (0/1). 

-  We consider using both Entropy and Information gain to choose on the
   attribute for splitting
-  Perform or Binary or Multiway Split
-  Use Techniques of Pruning, Regularization to stop splitting
-  Use 10/90 Split for test and training data
-  The training data is used with a 10 fold cross validation to choose model
   parameters

Implementation
--------------

Implemented the initial tree as follows 
---------------------------------------
- Implemented full decision tree
-Split on attribute with best entropy gain
- Stopping criteria either all in the same class or No attributes or no records left
- Naive assumption of uniform distribution for continuous data and used 10 uniform bins based on range
 
Results 1
---------
Accuracy result of this initial implementation (with cross validation)  
	- average with cross validation set - 0.7958
	- accuracy with the (10%) hold out test data set - 0.7916

Improved the Initial Tree as Follows 
------------------------------------
Dynamic KTile binning. K tuned by crossvalidation. KTile performed best of uniform, Gaussian
Pruned after splitting 
 Pruned subtrees with cost complexity is higher than parent node
 cross-validation trains cost complexity factor 
Further pruning 
 Removing one subtree at a time having the lowest improvement per leaf
 Cross validation selects the best tree

Results with KTile Binning
------------------------------
- Avergae on crossvalidation data set (10 fold)
	KTile=4,
	Accuracy= 0.8236

- On hold out test data set (10%)
	KTile=4,
	Accuracy= 0.8216


Results with KTile Binning and Cost Complexity Pruning
----------------------------------------------------------
- Avergae on crossvalidation data set (10 fold)
	KTile=2,
	costComplexityFactor=0.8
	Accuracy= 0.8435

- On hold out test data set (10%)
	KTile=2,
	costComplexityFactor=0.8
	Accuracy= 0.8344

- Best single Cross Validation
	KTile=2,
	costComplexityFactor=0.8
	Accuracy= 0.8662




1. This script requires python version 2.6.

2. I have tested the script on Ubuntu 14.04.3 LTS with Python 2.7.6

3. To execute the program to train based on the dataset execute
python tree.py

4.This will do the following 
 	4.1 perform cross validation and write the steps of validation to stdout and to file Dsouza-Ajay-result-crossvalidation.txt 

	4.2 select the best cross validated model and write avergae accuracy for the test set to file stdout and to file Dsouza-Ajay-result.txt 

	4.3 This will also save the best trained decision tree to disk as json in file decision_tree.json

	4.4 Training with 10 fold cross validation and pruning takes around 30 mins

5. To execute the classification and report averge on test set using the previously trained decision tree saved in file file decision_tree.json execute
python tree.py useSavedTree

6.To execute the trained decision tree saved in a different json file execute
python tree.py useSavedTree <json_file_name_to_read>

7. The best decision tree using cross validation uses
   - dynamic ktile with k = 4
   - Cost complexity tree pruning with complexity factor = 0.8
   - splitting based on entropy gai




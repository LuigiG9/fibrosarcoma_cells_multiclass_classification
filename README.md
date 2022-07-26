# fibrosarcoma_cells_multiclass_classification
Fibrosarcoma is a malignant tumour affecting the fibrous connective tissue. It is characterized by undifferentiated fibroblasts organized to form a stratiform pattern, usually appearing in males between 30 and 40 years. It originates in fibrous bone tissues and mainly spread in long or flat bones, often involving periosteum and overlying muscle. There are different types of diagnosable fibrosarcoma (e.g. melanoma of the spindle cells, synovial sarcoma, etc.), with traditional tests based on immunohistochemistry to exploit the selective identification of antibodies to antigens in biological tissues. Unfortunately, immunohistochemistry has some disadvantages, including among all i) costly equipment and ii) a high operator-dependant outcome. As a consequence, the analysis of the individual cells of the neoplastic tissue is preferable, as it allows us to evaluate in a more precise way cancer behavior or the response to therapy. Usually, these evaluations are made in laboratories by expert operators and require long times and high consumption of resources. The aim of this work is to recognise four different cell types (multi-class classification) according to their morphology.

This is an example of the images in the dataset:
![image](https://user-images.githubusercontent.com/58850870/181022007-97434060-7322-41b9-b210-cd778fda38e8.png)

Our workflow followed this stages:
1. Partitioning
2. Data Augmentation
3. Learning Process
4. Parameters Selection
5. Cross Validation
6. Results Evaluation

#  Partitioning

We've splitted the dataset into 90% ( Train set) and 10% (Local Test Set). The Training set has been splitted again into 80% (Training Set) and 20% (Validation Set)

#  Data Augmentation

We've used "ImageDataGenerator" to augment our dataset. Specifically, they've been implemented : Vertical and Horizontal flips, rotations ( with an angle from 0° to 90°), zoom ( with a range from 0.7 to 1.3) and a featurewise normalization.

#  Model
![image](https://user-images.githubusercontent.com/58850870/181023404-7e989e43-d79a-4cd3-b041-6c78f5990c8c.png)
In this picture it's possible to see our model. We've created a DNN that has the following structure until we achieve the last layers: 2 convolutional layers followed by a maxpooling layer ( with a factor of 2). 

![image](https://user-images.githubusercontent.com/58850870/181024177-109a5779-eda5-4057-a4d9-0ff885a26d56.png)
These were our results about accuracy and loss.

# Cross Validation

As it was said before, we've implemented a cross validation:
![image](https://user-images.githubusercontent.com/58850870/181024501-edb415af-190e-4508-90e5-b9fa743d3917.png)
At each loop, one folder is selected to be the 'local test' and another folder ( from the remaining ones) is selected to be the 'validation set'. 

# Results

We've obtained the following Confusion Matrix and an accuracy of 82%
![image](https://user-images.githubusercontent.com/58850870/181025285-b682cb9e-427d-4f28-aa77-52e3e9176534.png)

![image](https://user-images.githubusercontent.com/58850870/181025368-6937c61b-2a7d-43a0-9bc8-e2cf2920a3f0.png)




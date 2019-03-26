---
title: "HAR(Human Activity Recognition) - Predict Correct Method to Lift Dombell"
output:
  html_document: 
    keep_md: true
  pdf_document: default
  fig_height: 4
theme: spacelab
highlight: pygments
---



* * *


## Part 1: Setup
  
#### 1.1: Load packages
  

```r
Packages <- c("tidyverse", "ggplot2", "dplyr", "statsr", "GGally","caret","magrittr",  "purrr", "e1071", "rattle", "nnet", "rpart",  "ipred", "randomForest","gbm" )
lapply(Packages, library, character.only = TRUE)
```

Set seed for the project for reproducibility.

```r
set.seed(7777)
```

#### 1.2: Load Data

Data is provided in CSV format at the URL's provided below. I used `readr::read_csv` function to read the source training and testing data files.


```r
HAR_train_validate <- read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
HAR_testing <- read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
```

#### 1.3: Split Data into Training and Validation Sets
We will create training, validation and testing data sets partitions. We are already given a `testing` data set which we will use for final testing of our algorithm.

We will split our training data set into training and validation data sets by 75/25 split.


```r
trainIndex <- createDataPartition(HAR_train_validate$classe, 
                                  p = .75, 
                                  list = FALSE, 
                                  times = 1)

HAR_training <- HAR_train_validate[trainIndex,]
HAR_validate  <- HAR_train_validate[-trainIndex,]
```




* * *
  
  
  
  
## Part 2: Research Question

These days we can easily collect a large amount of data about different personal activities. In this project, six young healthy participants were asked to perform a set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different ways. We collected data by attaching accelerometers on following areas:

- Belt
- Forearm
- Arm
- Dumbbell

These 6 participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. These 5 different ways to lift barbell were recorded in `classe` variable. Given below is the detail of these labels in `classe` variable:


    Class A - Peformed exercise exactly according to the specification 
    Class B - Peformed exercise throwing the elbows to the front
    Class C - Peformed exercise lifting the dumbbell only halfway 
    Class D - Peformed exercise lowering the dumbbell only halfway
    Class E - Peformed exercise throwing the hips to the front



We will try to predict if a person lifted barbell correctly or incorrectly by predicting the `classe` variable by using other varibles in the dataset. We will use training data to train our model and then use validate data to measure the accuracy of our model. Finally we will do our prediction on 20 test cases provided to us in the project.


* * *
  
## Part 3: Exploratory Data Analysis and Data Prepration
  
#### 3.1: Glimpse the dataset  
First we will see the types of all variables provided in the movies data-set.


```r
#glimpse(HAR_training)
```

Complete output of `glimpse` is available in the appendix section. Here we will provide a quick summary of the variables. 

We notice that there are summary and non-summary variables in the dataset. Summary variables have NA values for most of the rows except few rows where they do provide summary of the other variables.
Following variables are non-summary variables that have non-NA values for most of the rows, other variables are summary variables that have mostly NA's in them:

    
    - roll_(Forearm/Belt/Arm/Dumbell)
    - pitch_(Forearm/Belt/Arm/Dumbell)
    - yaw_(Forearm/Belt/Arm/Dumbell)
    - total_accel_(Forearm/Belt/Arm/Dumbell)
    - gyros_(Forearm/Belt/Arm/Dumbell)_x
    - gyros_(Forearm/Belt/Arm/Dumbell)_y
    - gyross_(Forearm/Belt/Arm/Dumbell)_z
    - accel_(Forearm/Belt/Arm/Dumbell)_x
    - accel_(Forearm/Belt/Arm/Dumbell)_y
    - accel_(Forearm/Belt/Arm/Dumbell)_z
    - magnet_(Forearm/Belt/Arm/Dumbell)_x
    - magnet_(Forearm/Belt/Arm/Dumbell)_y
    - magnet_(Forearm/Belt/Arm/Dumbell)_z

It seems that we have similar list of summary and non-summary variables for Belt, Arm and Dumbbell accelerators.



#### 3.2: Data Cleanup and Set Correct Datatypes

Now we will make sure that we have correct data types for our columns in our dataset. 

We can see that some columns have data types `chr` where it should be set to different data types. This is due to some of the cell in that variable include "#DIV/0!" value. First we will clean the data by adding NA to the cells which have "#DIV/0!" value and then change the type of  column to appropriate data type instead of `chr`.


```r
HAR_training$cvtd_timestamp  <- as.POSIXct(HAR_training$cvtd_timestamp, format = "%d/%m/%Y %H:%M")
HAR_training$user_name  <- as.factor(HAR_training$user_name)
HAR_training$new_window  <- as.factor(HAR_training$new_window)
HAR_training$classe  <- as.factor(HAR_training$classe)
# Changing chr data types to numeric
HAR_training %<>%
     mutate_if(is.character,as.numeric)


#Perform same for vlidation dataset
HAR_validate$cvtd_timestamp  <- as.POSIXct(HAR_validate$cvtd_timestamp, format = "%d/%m/%Y %H:%M")
HAR_validate$user_name  <- as.factor(HAR_validate$user_name)
HAR_validate$new_window  <- as.factor(HAR_validate$new_window)
HAR_validate$classe  <- as.factor(HAR_validate$classe)
# Changing chr data types to numeric
HAR_validate %<>%
  mutate_if(is.character,as.numeric)


HAR_testing$cvtd_timestamp  <- as.POSIXct(HAR_testing$cvtd_timestamp, format = "%d/%m/%Y %H:%M")
HAR_testing$user_name  <- as.factor(HAR_testing$user_name)
HAR_testing$new_window  <- as.factor(HAR_testing$new_window)
#HAR_testing$classe  <- as.factor(HAR_testing$classe)
# Changing chr data types to numeric
HAR_testing %<>%
  mutate_if(is.character,as.numeric)
```



Next we will create a new dataset containing all non-summary variables for Belt, Arm and Dumbbell accelerators. This data set will be used for exploring these variables in more detail and also to create our predictive model.



```r
HAR_training_nonSummary <- HAR_training %>%
        select(classe,
               roll_forearm, pitch_forearm, yaw_forearm,total_accel_forearm,
               gyros_forearm_x, gyros_forearm_y, gyros_forearm_z,
               accel_forearm_x, accel_forearm_y, accel_forearm_z,
               magnet_forearm_x, magnet_forearm_y, magnet_forearm_z,
               
               roll_belt, pitch_belt, yaw_belt,total_accel_belt,
               gyros_belt_x, gyros_belt_y, gyros_belt_z,
               accel_belt_x, accel_belt_y, accel_belt_z,
               magnet_belt_x, magnet_belt_y, magnet_belt_z,
               
               roll_arm, pitch_arm, yaw_arm,total_accel_arm,
               gyros_arm_x, gyros_arm_y, gyros_arm_z,
               accel_arm_x, accel_arm_y, accel_arm_z,
               magnet_arm_x, magnet_arm_y, magnet_arm_z,
               
               roll_dumbbell, pitch_dumbbell, yaw_dumbbell,total_accel_dumbbell,
               gyros_dumbbell_x, gyros_dumbbell_y, gyros_dumbbell_z,
               accel_dumbbell_x, accel_dumbbell_y, accel_dumbbell_z,
               magnet_dumbbell_x, magnet_dumbbell_y, magnet_dumbbell_z
               
        )



HAR_validate_nonSummary <- HAR_validate %>%
        select(classe,
               roll_forearm, pitch_forearm, yaw_forearm,total_accel_forearm,
               gyros_forearm_x, gyros_forearm_y, gyros_forearm_z,
               accel_forearm_x, accel_forearm_y, accel_forearm_z,
               magnet_forearm_x, magnet_forearm_y, magnet_forearm_z,
               
               roll_belt, pitch_belt, yaw_belt,total_accel_belt,
               gyros_belt_x, gyros_belt_y, gyros_belt_z,
               accel_belt_x, accel_belt_y, accel_belt_z,
               magnet_belt_x, magnet_belt_y, magnet_belt_z,
               
               roll_arm, pitch_arm, yaw_arm,total_accel_arm,
               gyros_arm_x, gyros_arm_y, gyros_arm_z,
               accel_arm_x, accel_arm_y, accel_arm_z,
               magnet_arm_x, magnet_arm_y, magnet_arm_z,
               
               roll_dumbbell, pitch_dumbbell, yaw_dumbbell,total_accel_dumbbell,
               gyros_dumbbell_x, gyros_dumbbell_y, gyros_dumbbell_z,
               accel_dumbbell_x, accel_dumbbell_y, accel_dumbbell_z,
               magnet_dumbbell_x, magnet_dumbbell_y, magnet_dumbbell_z
               
        )



HAR_test_nonSummary <- HAR_testing %>%
        select(roll_forearm, pitch_forearm, yaw_forearm,total_accel_forearm,
               gyros_forearm_x, gyros_forearm_y, gyros_forearm_z,
               accel_forearm_x, accel_forearm_y, accel_forearm_z,
               magnet_forearm_x, magnet_forearm_y, magnet_forearm_z,
               
               roll_belt, pitch_belt, yaw_belt,total_accel_belt,
               gyros_belt_x, gyros_belt_y, gyros_belt_z,
               accel_belt_x, accel_belt_y, accel_belt_z,
               magnet_belt_x, magnet_belt_y, magnet_belt_z,
               
               roll_arm, pitch_arm, yaw_arm,total_accel_arm,
               gyros_arm_x, gyros_arm_y, gyros_arm_z,
               accel_arm_x, accel_arm_y, accel_arm_z,
               magnet_arm_x, magnet_arm_y, magnet_arm_z,
               
               roll_dumbbell, pitch_dumbbell, yaw_dumbbell,total_accel_dumbbell,
               gyros_dumbbell_x, gyros_dumbbell_y, gyros_dumbbell_z,
               accel_dumbbell_x, accel_dumbbell_y, accel_dumbbell_z,
               magnet_dumbbell_x, magnet_dumbbell_y, magnet_dumbbell_z
               
        )
```

#### 3.3: Remove NA Values

As NA values can cause difficulties with our modeling algorithm, we will remove any rows with NA values in our non-summary data sets.


```r
# Drop rows with NA's in HAR_training_nonSummary
HAR_training_nonSummary <- HAR_training_nonSummary %>% drop_na()
colnames(HAR_training_nonSummary)[colSums(is.na(HAR_training_nonSummary)) > 0]
```

```
## character(0)
```
There are no rows with NA values left in HAR_training_nonSummary data set.



```r
# Drop rows with NA's in HAR_validate_nonSummary
HAR_validate_nonSummary <- HAR_validate_nonSummary %>% drop_na()
colnames(HAR_validate_nonSummary)[colSums(is.na(HAR_validate_nonSummary)) > 0]
```

```
## character(0)
```



```r
# Drop rows with NA's in HAR_validate_nonSummary
HAR_test_nonSummary <- HAR_test_nonSummary %>% drop_na()
colnames(HAR_test_nonSummary)[colSums(is.na(HAR_test_nonSummary)) > 0]
```

```
## character(0)
```



Also, There are no rows with NA values left in HAR_validate_nonSummary data set.


#### 3.4: Data Distributions

Distrubutions for all non-summary varibales are provided in the appendix section.



#### 3.5: Outliers
After looking at box plots in the appendix section, we don’t need to remove any outliers from our dataset.



#### 3.6: Data Normlization
We don’t not need to normalize our data as we will be using tree based algorithm such for our modeling and predictions.

* * *



## Part 4: Modeling


We built multiple models using different algorithms with non-summary variables. We then compared the accuracies of our different models using validation data sets. Finally we picked the model with the best accuracy to be used as our final model to predict our test dataset.

We built following models and then compared their accuracies:


- Model A - Tree algorithm using non-summary variables	**Accuracy : 0.74**

- Model B - Algorithm using Principal Components Analysis with non-summary variables	**Accuracy : 0.94**

- Model C - RandomForest algorithm using non-summary variables	**Accuracy : 0.989**

- Model D - Multinomial Logistic Regression algorithm using non-summary variables	**Accuracy : 0.67**

- Model E - Tree algorithm using non-summary variables wth Grid hyper parameters	**Accuracy : 0.74**

- Model F - Bagged Tree model using non-summary Variables **Accuracy : 0.988**

- Model G - Cross Validation using K-fold with Decision Tree Algorithm	**Accuracy : 0.54, 0.41, 0.31**

- Model H - Gradiant Boosting Machine (GBM) Tree Algorithm using non-summary variables	**Accuracy : 0.94**



Based on these accuracies I decided to use Random Forest algorithm using non-summary variables which produced Accuracy : 0.989.


### 4.1: Model C - RandomForest algorithm using non-summary variables

Next we will try RandomForest algorithm with `ntree = 100` to fit our model and do predictions on our validation and test data sets.


```r
#Model - C Non-Summary Variables Randomforest model

#modelFit_C <- train(classe ~ ., data = HAR_training_nonSummary, method = "rf", ntree = 10)
modelFit_C <- randomForest(classe ~ ., data = HAR_training_nonSummary, method = "rf", ntree = 100)
prediction_C <- predict(modelFit_C, HAR_validate_nonSummary)
confusionMatrix(prediction_C, HAR_validate_nonSummary$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    5    0    0    0
##          B    0  943    3    0    0
##          C    0    1  852    5    1
##          D    0    0    0  799    1
##          E    0    0    0    0  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9967          
##                  95% CI : (0.9947, 0.9981)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9959          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9937   0.9965   0.9938   0.9978
## Specificity            0.9986   0.9992   0.9983   0.9998   1.0000
## Pos Pred Value         0.9964   0.9968   0.9919   0.9988   1.0000
## Neg Pred Value         1.0000   0.9985   0.9993   0.9988   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1923   0.1737   0.1629   0.1833
## Detection Prevalence   0.2855   0.1929   0.1752   0.1631   0.1833
## Balanced Accuracy      0.9993   0.9965   0.9974   0.9968   0.9989
```

```r
#print teh model and look at its output
print(modelFit_C)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = HAR_training_nonSummary,      method = "rf", ntree = 100) 
##                Type of random forest: classification
##                      Number of trees: 100
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.58%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4181    4    0    0    0 0.0009557945
## B   16 2826    6    0    0 0.0077247191
## C    0   19 2544    4    0 0.0089598753
## D    0    0   25 2384    3 0.0116086235
## E    0    0    3    5 2698 0.0029563932
```


### 4.2: Out Of Bag Error Rates



```r
# Look at final OOB error rate (last row in err matrix)
err <- modelFit_C$err.rate
oob_err <- err[nrow(err), "OOB"]
print(oob_err)
```

```
##         OOB 
## 0.005775241
```

```r
# Plot the model trained in the previous exercise
plot(modelFit_C)
legend(x = "right", 
       legend = colnames(err),
       fill = 1:ncol(err))
```

![](JH-Course8-FinalPrfoject_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

Our lowest OOB error rate is **0.0057**. Also, we can see that Out of Bag error (OOB) becomes flat after 60 trees. So we can use 60 as number `ntrees` of trees for our algorithm.



### 4.3: Fine Tuning Random Forest Algorithm Parameters

Next we will try to fine tune our Random Forest model by finding optimal values for parameters such as `mtry`, `nodesize`, `sampsize` using grid parameters. i.e. we will try multiple combinations of these values to find the combination which will result in the highest model accuracy.



```r
#########################################################
########### Tune all of the mtry, nodesize, sampsize ####
#########################################################

# Establish a list of possible values for mtry, nodesize and sampsize
#mtry <- seq(4, ncol(HAR_training_nonSummary) * 0.8, 2)
mtry <- 7
nodesize <- seq(3, 8, 2)
#nodesize <- 3
sampsize <- nrow(HAR_training_nonSummary) * c(0.7, 0.8)

# Create a data frame containing all combinations 
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, sampsize = sampsize)

# Create an empty vector to store OOB error values
oob_err <- c()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_grid)) {
  
  # Train a Random Forest model
  model <- randomForest(formula = classe ~ ., 
                        data = HAR_training_nonSummary,
                        ntree = 60, 
                        mtry = hyper_grid$mtry[i],
                        nodesize = hyper_grid$nodesize[i],
                        sampsize = hyper_grid$sampsize[i])
  
  # Store OOB error for the model                      
  oob_err[i] <- model$err.rate[nrow(model$err.rate), "OOB"]
}

# Identify optimal set of hyperparmeters based on OOB error
opt_i <- which.min(oob_err)
print(hyper_grid[opt_i,])
```

```
##   mtry nodesize sampsize
## 1    7        3  10302.6
```


For tuning of our RandomForest we can use the following values to produce the best fit model:

mtry | nodesize |sampsize
---- | ---------|--------
7    |   3      |11774
   


Also the default value of mtry for random forests is square root of total number of features (for classification) and number of features divided by 3 for regression. We can see that `mtry = 7` in our RF would be most optimal to minimize OOB error. As default value for mtry would be 7 as well, so we can leave the default value for mtry.

We can see that this model's accuracy is quite high with **Accuracy : 0.98**. There are very few false classification for different class labels for our response variable `classe`.

### 4.4: Cross Validation using K-fold with RandomForest Tree Algorithm

Next we will try cross validation using K-fold with using random forest Algorithm to see if we can get a more accurate measure for our accuracy.



```r
train_control <- trainControl(method = "cv", number = 3)   # Cross-validation, 3 K-folds
modelFit_G <- train(classe ~ ., data = HAR_training_nonSummary, trControl=train_control, method = "rf", ntree = 80)
print(modelFit_G)
```

```
## Random Forest 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 9811, 9813, 9812 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9881098  0.9849557
##   27    0.9896043  0.9868483
##   52    0.9838970  0.9796286
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```


We can see that with 3-K cross validation, our accuracy is **Accuracy : 0.98**. There are very few false classification for different class labels for our response variable `classe`.



## Part 5: Predicting Test Data


```r
prediction_test <- predict(modelFit_C, HAR_test_nonSummary)
print(prediction_test)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

* * *



## Apendix



```r
glimpse(HAR_training)
```

```
## Observations: 14,718
## Variables: 160
## $ X1                       <dbl> 1, 2, 3, 4, 7, 9, 10, 11, 12, 13, 15, 1…
## $ user_name                <fct> carlitos, carlitos, carlitos, carlitos,…
## $ raw_timestamp_part_1     <dbl> 1323084231, 1323084231, 1323084231, 132…
## $ raw_timestamp_part_2     <dbl> 788290, 808298, 820366, 120339, 368296,…
## $ cvtd_timestamp           <dttm> 2011-12-05 11:23:00, 2011-12-05 11:23:…
## $ new_window               <fct> no, no, no, no, no, no, no, no, no, no,…
## $ num_window               <dbl> 11, 11, 11, 12, 12, 12, 12, 12, 12, 12,…
## $ roll_belt                <dbl> 1.41, 1.41, 1.42, 1.48, 1.42, 1.43, 1.4…
## $ pitch_belt               <dbl> 8.07, 8.07, 8.07, 8.05, 8.09, 8.16, 8.1…
## $ yaw_belt                 <dbl> -94.4, -94.4, -94.4, -94.4, -94.4, -94.…
## $ total_accel_belt         <dbl> 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, …
## $ kurtosis_roll_belt       <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ kurtosis_picth_belt      <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ kurtosis_yaw_belt        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ skewness_roll_belt       <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ skewness_roll_belt.1     <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ skewness_yaw_belt        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ max_roll_belt            <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ max_picth_belt           <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ max_yaw_belt             <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ min_roll_belt            <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ min_pitch_belt           <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ min_yaw_belt             <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ amplitude_roll_belt      <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ amplitude_pitch_belt     <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ amplitude_yaw_belt       <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ var_total_accel_belt     <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ avg_roll_belt            <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ stddev_roll_belt         <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ var_roll_belt            <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ avg_pitch_belt           <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ stddev_pitch_belt        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ var_pitch_belt           <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ avg_yaw_belt             <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ stddev_yaw_belt          <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ var_yaw_belt             <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ gyros_belt_x             <dbl> 0.00, 0.02, 0.00, 0.02, 0.02, 0.02, 0.0…
## $ gyros_belt_y             <dbl> 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0…
## $ gyros_belt_z             <dbl> -0.02, -0.02, -0.02, -0.03, -0.02, -0.0…
## $ accel_belt_x             <dbl> -21, -22, -20, -22, -22, -20, -21, -21,…
## $ accel_belt_y             <dbl> 4, 4, 5, 3, 3, 2, 4, 2, 2, 4, 2, 4, 4, …
## $ accel_belt_z             <dbl> 22, 22, 23, 21, 21, 24, 22, 23, 23, 21,…
## $ magnet_belt_x            <dbl> -3, -7, -2, -6, -4, 1, -3, -5, -2, -3, …
## $ magnet_belt_y            <dbl> 599, 608, 600, 604, 599, 602, 609, 596,…
## $ magnet_belt_z            <dbl> -313, -311, -305, -310, -311, -312, -30…
## $ roll_arm                 <dbl> -128, -128, -128, -128, -128, -128, -12…
## $ pitch_arm                <dbl> 22.5, 22.5, 22.5, 22.1, 21.9, 21.7, 21.…
## $ yaw_arm                  <dbl> -161, -161, -161, -161, -161, -161, -16…
## $ total_accel_arm          <dbl> 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,…
## $ var_accel_arm            <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ avg_roll_arm             <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ stddev_roll_arm          <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ var_roll_arm             <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ avg_pitch_arm            <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ stddev_pitch_arm         <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ var_pitch_arm            <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ avg_yaw_arm              <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ stddev_yaw_arm           <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ var_yaw_arm              <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ gyros_arm_x              <dbl> 0.00, 0.02, 0.02, 0.02, 0.00, 0.02, 0.0…
## $ gyros_arm_y              <dbl> 0.00, -0.02, -0.02, -0.03, -0.03, -0.03…
## $ gyros_arm_z              <dbl> -0.02, -0.02, -0.02, 0.02, 0.00, -0.02,…
## $ accel_arm_x              <dbl> -288, -290, -289, -289, -289, -288, -28…
## $ accel_arm_y              <dbl> 109, 110, 110, 111, 111, 109, 110, 110,…
## $ accel_arm_z              <dbl> -123, -125, -126, -123, -125, -122, -12…
## $ magnet_arm_x             <dbl> -368, -369, -368, -372, -373, -369, -37…
## $ magnet_arm_y             <dbl> 337, 337, 344, 344, 336, 341, 334, 339,…
## $ magnet_arm_z             <dbl> 516, 513, 513, 512, 509, 518, 516, 509,…
## $ kurtosis_roll_arm        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ kurtosis_picth_arm       <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ kurtosis_yaw_arm         <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ skewness_roll_arm        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ skewness_pitch_arm       <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ skewness_yaw_arm         <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ max_roll_arm             <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ max_picth_arm            <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ max_yaw_arm              <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ min_roll_arm             <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ min_pitch_arm            <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ min_yaw_arm              <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ amplitude_roll_arm       <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ amplitude_pitch_arm      <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ amplitude_yaw_arm        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ roll_dumbbell            <dbl> 13.05217, 13.13074, 12.85075, 13.43120,…
## $ pitch_dumbbell           <dbl> -70.49400, -70.63751, -70.27812, -70.39…
## $ yaw_dumbbell             <dbl> -84.87394, -84.71065, -85.14078, -84.87…
## $ kurtosis_roll_dumbbell   <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ kurtosis_picth_dumbbell  <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ kurtosis_yaw_dumbbell    <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ skewness_roll_dumbbell   <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ skewness_pitch_dumbbell  <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ skewness_yaw_dumbbell    <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ max_roll_dumbbell        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ max_picth_dumbbell       <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ max_yaw_dumbbell         <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ min_roll_dumbbell        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ min_pitch_dumbbell       <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ min_yaw_dumbbell         <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ amplitude_roll_dumbbell  <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ amplitude_pitch_dumbbell <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ amplitude_yaw_dumbbell   <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ total_accel_dumbbell     <dbl> 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,…
## $ var_accel_dumbbell       <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ avg_roll_dumbbell        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ stddev_roll_dumbbell     <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ var_roll_dumbbell        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ avg_pitch_dumbbell       <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ stddev_pitch_dumbbell    <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ var_pitch_dumbbell       <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ avg_yaw_dumbbell         <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ stddev_yaw_dumbbell      <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ var_yaw_dumbbell         <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ gyros_dumbbell_x         <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
## $ gyros_dumbbell_y         <dbl> -0.02, -0.02, -0.02, -0.02, -0.02, -0.0…
## $ gyros_dumbbell_z         <dbl> 0.00, 0.00, 0.00, -0.02, 0.00, 0.00, 0.…
## $ accel_dumbbell_x         <dbl> -234, -233, -232, -232, -232, -232, -23…
## $ accel_dumbbell_y         <dbl> 47, 47, 46, 48, 47, 47, 48, 47, 47, 48,…
## $ accel_dumbbell_z         <dbl> -271, -269, -270, -269, -270, -269, -27…
## $ magnet_dumbbell_x        <dbl> -559, -555, -561, -552, -551, -549, -55…
## $ magnet_dumbbell_y        <dbl> 293, 296, 298, 303, 295, 292, 291, 299,…
## $ magnet_dumbbell_z        <dbl> -65, -64, -63, -60, -70, -65, -69, -64,…
## $ roll_forearm             <dbl> 28.4, 28.3, 28.3, 28.1, 27.9, 27.7, 27.…
## $ pitch_forearm            <dbl> -63.9, -63.9, -63.9, -63.9, -63.9, -63.…
## $ yaw_forearm              <dbl> -153, -153, -152, -152, -152, -152, -15…
## $ kurtosis_roll_forearm    <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ kurtosis_picth_forearm   <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ kurtosis_yaw_forearm     <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ skewness_roll_forearm    <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ skewness_pitch_forearm   <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ skewness_yaw_forearm     <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ max_roll_forearm         <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ max_picth_forearm        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ max_yaw_forearm          <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ min_roll_forearm         <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ min_pitch_forearm        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ min_yaw_forearm          <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ amplitude_roll_forearm   <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ amplitude_pitch_forearm  <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ amplitude_yaw_forearm    <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ total_accel_forearm      <dbl> 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,…
## $ var_accel_forearm        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ avg_roll_forearm         <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ stddev_roll_forearm      <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ var_roll_forearm         <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ avg_pitch_forearm        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ stddev_pitch_forearm     <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ var_pitch_forearm        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ avg_yaw_forearm          <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ stddev_yaw_forearm       <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ var_yaw_forearm          <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
## $ gyros_forearm_x          <dbl> 0.03, 0.02, 0.03, 0.02, 0.02, 0.03, 0.0…
## $ gyros_forearm_y          <dbl> 0.00, 0.00, -0.02, -0.02, 0.00, 0.00, 0…
## $ gyros_forearm_z          <dbl> -0.02, -0.02, 0.00, 0.00, -0.02, -0.02,…
## $ accel_forearm_x          <dbl> 192, 192, 196, 189, 195, 193, 190, 193,…
## $ accel_forearm_y          <dbl> 203, 203, 204, 206, 205, 204, 205, 205,…
## $ accel_forearm_z          <dbl> -215, -216, -213, -214, -215, -214, -21…
## $ magnet_forearm_x         <dbl> -17, -18, -18, -16, -18, -16, -22, -17,…
## $ magnet_forearm_y         <dbl> 654, 661, 658, 658, 659, 653, 656, 657,…
## $ magnet_forearm_z         <dbl> 476, 473, 469, 469, 470, 476, 473, 465,…
## $ classe                   <fct> A, A, A, A, A, A, A, A, A, A, A, A, A, …
```





```r
ggpairs(HAR_training_nonSummary, columns = c("classe"))
```

![](JH-Course8-FinalPrfoject_files/figure-html/unnamed-chunk-16-1.png)<!-- -->

```r
ggpairs(HAR_training_nonSummary, columns = c("classe", "roll_belt", "pitch_forearm"))
```

![](JH-Course8-FinalPrfoject_files/figure-html/unnamed-chunk-16-2.png)<!-- -->








```r
HAR_training_nonSummary %>%
                select(c(2:12)) %>%
                gather() %>%                              # Convert to key-value pairs
                ggplot(aes(value)) +                    # Plot the values
                facet_wrap(~ key, scales = "free") +      # In separate panels
                geom_density()                           # as density
```

![](JH-Course8-FinalPrfoject_files/figure-html/unnamed-chunk-17-1.png)<!-- -->

```r
HAR_training_nonSummary %>%
                select(c(13:26)) %>%
                gather() %>%                              # Convert to key-value pairs
                ggplot(aes(value)) +                      # Plot the values
                facet_wrap(~ key, scales = "free") +      # In separate panels
                geom_density()                            # as density
```

![](JH-Course8-FinalPrfoject_files/figure-html/unnamed-chunk-17-2.png)<!-- -->

```r
HAR_training_nonSummary %>%
                select(c(27:39)) %>%
                gather() %>%                              # Convert to key-value pairs
                ggplot(aes(value)) +                      # Plot the values
                facet_wrap(~ key, scales = "free") +      # In separate panels
                geom_density()                            # as density
```

![](JH-Course8-FinalPrfoject_files/figure-html/unnamed-chunk-17-3.png)<!-- -->

```r
HAR_training_nonSummary %>%
                select(c(40:52)) %>%
                gather() %>%                              # Convert to key-value pairs
                ggplot(aes(value)) +                      # Plot the values
                facet_wrap(~ key, scales = "free") +      # In separate panels
                geom_density()                            # as density
```

![](JH-Course8-FinalPrfoject_files/figure-html/unnamed-chunk-17-4.png)<!-- -->


#### Outliers
We will draw boxplots of our non-summary variables to see if there are any outliers in the dataset and if they need to be removed ahead of our modeling exercise.


```r
#######################################################################################

HAR_training_nonSummary %>%
                select(c(2:12)) %>%
                gather() %>%                              # Convert to key-value pairs
                ggplot(aes(y = value)) +                    # Plot the values
                facet_wrap(~ key, scales = "free") +      # In separate panels
#               geom_density()                           # as density
                geom_boxplot()                            # as boxplot
```

![](JH-Course8-FinalPrfoject_files/figure-html/unnamed-chunk-18-1.png)<!-- -->

```r
HAR_training_nonSummary %>%
                select(c(13:26)) %>%
                gather() %>%                              # Convert to key-value pairs
                ggplot(aes(y = value)) +                    # Plot the values
                facet_wrap(~ key, scales = "free") +      # In separate panels
#                geom_density()                           # as density
                geom_boxplot() 
```

![](JH-Course8-FinalPrfoject_files/figure-html/unnamed-chunk-18-2.png)<!-- -->

```r
HAR_training_nonSummary %>%
                select(c(27:39)) %>%
                gather() %>%                              # Convert to key-value pairs
                ggplot(aes(y = value)) +                    # Plot the values
                facet_wrap(~ key, scales = "free") +      # In separate panels
#                geom_density()                           # as density
                geom_boxplot() 
```

![](JH-Course8-FinalPrfoject_files/figure-html/unnamed-chunk-18-3.png)<!-- -->

```r
HAR_training_nonSummary %>%
                select(c(40:52)) %>%
                gather() %>%                              # Convert to key-value pairs
                ggplot(aes(y = value)) +                    # Plot the values
                facet_wrap(~ key, scales = "free") +      # In separate panels
#                geom_density()                           # as density
                geom_boxplot() 
```

![](JH-Course8-FinalPrfoject_files/figure-html/unnamed-chunk-18-4.png)<!-- -->







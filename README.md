# House-Price-Prediction

### Objective:
To develop a machine learning model that predicts the final sale price of homes based on housing features. 

**keywords:** supervised learning, regression 

### Evaluation Metric: RMSLE
- handles large variations in house prices
- ensures errors are considered in terms of relative size (percentage errors)
- dealing with prices, percentage error is crucial

## EDA Process
### Goal:
1. Understand the variables we are working with (numerical, categorical, missing values, etc)
2. Understand the target variable 'SalePrice' (distribution, skewness)
3. Find features that are correlated with 'SalePrice'

### Handling Missing Data
| Variable Type  | Strategy |
| ------------- | ------------- |
| Numerical  | Fill with median or model-based  |
| Categorical (really missing)  | Fill with mode  |
| Categorical (NA represent 0) | Replace NA |
| Large Missing Values | Consider dropping |

Variables who are missing >80% consider dropping them, missing <5% consider filling with median/mode, and <1% consider dropping observation.

### Feature Engineering
Added variables that provide more information and could be crucial in the model. 'TotalSF' tracks the total space. 'AgeAtSale' shows the age of the house at the sale date. 'GarageAge' shows the age of the garage at the sale date. 'TotalBathrooms' tracks the total number of bathrooms. Other variables are important indicators.

### Transformation 
Due to the right skewness and the fact that we are working with price change, it is important to capture percentage change. I use a log transform on 'SalePrice'.

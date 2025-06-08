# House-Price-Prediction
An end-to-end machine learning pipeline to predict home sale prices using feature engineering, regression models, and XGBoost.

Dataset: [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

### Objective:
To develop a machine learning model that predicts the final sale price of homes based on housing features. 

**keywords:** supervised learning, regression, random forest, XGBoost

### Evaluation Metric: RMSLE
RMSLE (Root Mean Squared Log Error) was chosen as the evaluation metric because it penalizes large underpredictions more than small overpredictions and handles wide price ranges better. It reflects relative error (percentage difference), which is more meaningful in pricing contexts.

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

For variables with >80% missing values consider dropping them, missing <5% consider filling with median/mode, and <1% consider dropping observation.

### Feature Engineering
Added variables that provide potential predictive information. 
- 'TotalSF' = tracks the total square footage (basement + living area)
-  'AgeAtSale' = shows the age of the house at the time of sale
-  'GarageAge' = shows the age of the garage at the time of sale
-  'TotalBathrooms' tracks the total number of bathrooms 
-  Other variables are important indicators.

### Transformation 
Due to the right skewness and the fact that we are working with price change, it is important to capture percentage change. I use a log transform on 'SalePrice'.

## Baseline Models
### Linear Regression
I used an LM model as a baseline because it is the most interpretable model, quick to train, and easy to debug. It gives us an idea of what features are significant and whether our engineered features are useful. LM model provides a baseline with performance, which I can use to compare with more complex models that should outperform simple regression.
**Note:** Conventional practice used the workflow pipeline in R. However, I ran into an issue where the preprocessing of the training data created dummy categorical variables that did not exist in the test data. Hence, leading to NA predictions. The fix is to manually prep() the test data, too. 

### Random Forest
I used an RF model as a baseline because it serves as a good nonlinear model, to begin with and compare its performance to the more interpretable models. In addition, it serves as a strong tree-based benchmark and helps us determine if there are interactions between the features that our linear model isn't capturing. A huge improvement in results would suggest that there is a nonlinearity or interactions that are not captured by simple regression.

**Note:** In sum, the LM model serves as an interpretable benchmark, and the RF Model gives more power due to its adaptability. 

### Results
| Evaluation |LM  | RF |
| ------------- | ------------- | ------------- |
| RMSE  | 0.162  | 0.162 |
| R^2 | 0.856 | 0.860 |
| RMSLE | 0.162 | 0.162 |

Predictions differ from actual observations on average by 0.162, which are great results for price prediction (<0.2). Our models explain for about 85% variation. To conclude, the features we selected are strong predictors. Since there is little improvement in the R^2, the problem appears mostly linear. Lastly, our baseline models seem to be near the performance ceiling.

## Fine Tune Advance Model
### XGBoost
We utilize a complex model that generalizes effectively to confirm if our baseline regression really is performing at its ceiling. Random forest indicates that there could be some nonlinearity or interactions since it performed slightly better than linear regression. XGBoost helps us conclude this behavior.

### Results
| Evaluation  | XGB |
| ------------- | ------------- |
| RMSE  | 0.148  |
| R^2 | 0.878 | 
| RMSLE | 0.148 | 

As we can see, the XGB model performed much better showing us that there were small but meaningful nonlinear structures not being captured. In addition, we used a variable importance plot ('vip') to identify which features the model relies on the most. Using the plot, we found that 'TotalSF', 'OverallQual', and 'GrLivArea' were the most important features. This aligns with our domain intuition and EDA, validating that our engineered variables improved model performance.

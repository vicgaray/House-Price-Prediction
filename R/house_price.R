library(tidyverse)

# Data
train <- read_csv("../data/train.csv")
test <- read_csv("../data/test.csv")

########################## EDA Process
## (1)
dim(train)
glimpse(train)

#Count missing values in features
missing_data <- train %>%
  summarise_all(~ sum(is.na(.))) %>%
  pivot_longer(cols=everything(), names_to="variable", values_to="missing") %>%
  filter(missing>0) %>%
  arrange(desc(missing))

missing_data
missing_data %>%
  mutate(percent_missing = missing/nrow(train)) %>%
  arrange(desc(percent_missing))

## (2)
summary(train$SalePrice)
ggplot(train, aes(SalePrice)) +
  geom_histogram(bins=30, fill="steelblue", color="white") + 
  labs(title="SalePrice Distribution", x="Sale Price", y="Frequency")
# Note: SalePrice is rightly skewed containing high outliers.

## (3)
# Look for correlation
numeric_vars <- train %>%
  select(where(is.numeric))

corr <- cor(numeric_vars, use="pairwise.complete.obs")
corr_with_price <- sort(corr[, "SalePrice"], decreasing = TRUE)

head(corr_with_price, 15)

# Plot some of the highly correlated values
ggplot(train, aes(x=OverallQual, y=SalePrice)) +
  geom_point(alpha=0.5) +
  geom_smooth(method="lm") +
  labs(title="Material and Finish Quality vs Sale Price", x="Overall Material and Finish Quality", y="Sale Price")
# Note: This feature is ordinal categorical, but is treated as numeric. Hence, the unusual scatter pattern.

ggplot(train, aes(x=GrLivArea, y=SalePrice)) +
  geom_point(alpha=0.5) +
  geom_smooth(method="lm") +
  labs(title="Living Area vs Sale Price", x="Above Ground Living Area", y="Sale Price")

ggplot(train, aes(x=TotalBsmtSF, y=SalePrice)) +
  geom_point(alpha=0.5) +
  geom_smooth(method="lm") +
  labs(title="Total Basement Area vs Sale Price", x="Total Square Feet of Basement Area", y="Sale Price")

# Addressing Missing features
# Note: Ignore PoolQC, MiscFeature, Alley, Fence because they are missing 80+% of values
unique(train$FireplaceQu)
train$FireplaceQu[is.na(train$FireplaceQu)] <- "None"

unique(train$LotFrontage)
train$LotFrontage[is.na(train$LotFrontage)] <- median(train$LotFrontage, na.rm=TRUE)

unique(train$GarageYrBlt)
train$GarageYrBlt[is.na(train$GarageYrBlt)] <- median(train$GarageYrBlt, na.rm=TRUE)

unique(train$MasVnrArea) # <1% consider omitting obs
train$MasVnrArea[is.na(train$MasVnrArea)] <- 0

# Feature Engineering  
train <- train %>%
  mutate(
    TotalSF = TotalBsmtSF + `1stFlrSF` + `2ndFlrSF`, # Total space
    AgeAtSale = YrSold - YearBuilt, # Shows aging of house
    Remodeled = as.integer(YearRemodAdd != YearBuilt), # 1 - if the remodel date happened, 0 - if remodel didn't happen same date as built date
    HasBasement = as.integer(TotalBsmtSF>0), # 1 - if basement, 0 - if no basement
    GarageAge = YrSold - GarageYrBlt,
    HasGarage = as.integer(!is.na(GarageYrBlt)), 
    TotalBathrooms = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
  )

test <- test %>%
  mutate(
    TotalSF = TotalBsmtSF + `1stFlrSF` + `2ndFlrSF`, # Total space
    AgeAtSale = YrSold - YearBuilt, # Shows aging of house
    Remodeled = as.integer(YearRemodAdd != YearBuilt), # 1 - if the remodel date happened, 0 - if remodel didn't happen same date as built date
    HasBasement = as.integer(TotalBsmtSF>0), # 1 - if basement, 0 - if no basement
    GarageAge = YrSold - GarageYrBlt,
    HasGarage = as.integer(!is.na(GarageYrBlt)), 
    TotalBathrooms = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
  )

# Target Transformation
library(e1071)
skewness(train$SalePrice)

train <- train %>%
  mutate(SalePriceLog = log1p(SalePrice)) # log(x+1) to avoid x=0

ggplot(train, aes(SalePriceLog)) +
  geom_histogram(bins=30, fill="darkgreen", color="white") + 
  labs(title="Log-Transform SalePrice", x="Log Sale Price", y="Frequency")

numeric_vars <- numeric_vars %>%
  select(-Id, -SalePrice)
skew_vals <- sapply(numeric_vars, skewness, na.rm=TRUE)
sort(skew_vals[abs(skew_vals) > 0.75], decreasing = TRUE)

####################### Prepare Model & First Model
model_data <- train %>%
  select(
    SalePriceLog,
    OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, TotalSF,
    AgeAtSale, Remodeled, HasBasement, HasGarage, GarageAge, Neighborhood,
    HouseStyle, ExterQual, BsmtQual, KitchenQual
  )

# Split
set.seed(123)
train_index <- sample(1:nrow(model_data), 0.8 * nrow(model_data))
train_data <- model_data[train_index, ] 
val_data <- model_data[-train_index, ] 

################## Baseline Models
# Simple Linear Regression
linear_rec <- recipe(SalePriceLog ~., data=train_data) %>%
  step_string2factor(all_nominal_predictors()) %>% # Makes character columns into proper factors
  step_unknown(all_nominal_predictors(), new_level = "Missing") %>% # Address those houses without BsmtQual
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% # Addresses Categorical variables
  prep()

lm_train_processed <- bake(linear_rec, new_data = NULL)
lm_val_processed <- bake(linear_rec, new_data = val_data)

lm_fit <- linear_reg() %>% 
  set_engine("lm") %>%
  fit(SalePriceLog ~ ., data=lm_train_processed)

lm_preds <- predict(lm_fit, lm_val_processed) %>%
  bind_cols(truth=lm_val_processed$SalePriceLog)

metrics(lm_preds, truth=truth, estimate=.pred)
# Note: workflow is more conventionaly but test_data is not prepped
#   so I got NA values on prediction due to some categorical variables not
#   being dummied for the test data. We fix this by manually processing the
#   test data with recipe, step_dummy, prep, bake.

# Random Forest
rf_rec <- recipe(SalePriceLog ~., data=train_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  prep()

rf_train_processed <- bake(rf_rec, new_data = NULL)
rf_val_processed  <- bake(rf_rec, new_data = val_data)

rf_model <- rand_forest(trees=500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

rf_fit <- rf_model %>% fit(SalePriceLog ~., data=rf_train_processed)

rf_preds <- predict(rf_fit, new_data = rf_val_processed) %>%
  bind_cols(truth=rf_val_processed$SalePriceLog)

metrics(rf_preds, truth=truth, estimate=.pred)

# RMSLE
rmsle <- function(preds, actual) {
  preds[preds<0] <- 0 # Prevent negative predictions (lm)
  sqrt(mean((log1p(preds) - log1p(actual))^2))
}

rmsle_preds_lm <- expm1(lm_preds$.pred)
rmsle_actual_lm <- expm1(lm_preds$truth)
rmsle(rmsle_preds_lm, rmsle_actual_lm)

rmsle_preds_rf <- expm1(rf_preds$.pred)
rmsle_actual_rf <- expm1(rf_preds$truth)
rmsle(rmsle_preds_rf, rmsle_actual_rf)

############################# Select & Tune Model
# XGBoost
library(tidymodels)
library(xgboost)
library(doParallel)

xgb_rec <- recipe(SalePriceLog ~ ., data = train_data) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "Missing") %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  prep()

xgb_train_processed <- bake(xgb_rec, new_data = NULL)
xgb_val_processed  <- bake(xgb_rec, new_data = val_data)

xgb_spec <- boost_tree(
  trees = 1000, # max number of tree
  tree_depth = tune(), # how deep each tree can grow
  learn_rate = tune(), # how fast it learns 
  mtry = tune(), # number of predictors sampled
  loss_reduction = tune(), # gamma: how much a split needs to reduce loss
  sample_size = tune() # subsample ratio
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

# CV
set.seed(42)
cv_folds <- vfold_cv(xgb_train_processed, v=5)

xgb_wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_formula(SalePriceLog ~.)

xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  learn_rate(),
  mtry(range=c(5, ncol(xgb_train_processed)-1)),
  loss_reduction(),
  sample_size = sample_prop(),
  size = 20
)

registerDoParallel()

xgb_res <- tune_grid(
  xgb_wf,
  resamples = cv_folds,
  grid = xgb_grid,
  metrics = metric_set(rmse, rsq, mae),
  control = control_grid(save_pred=TRUE)
)

best_xgb <- select_best(xgb_res, metric="rmse")
best_xgb

final_xgb_wf <- finalize_workflow(xgb_wf, best_xgb)

final_xgb_fit <- final_xgb_wf %>% fit(data=xgb_train_processed)

xgb_preds <- predict(final_xgb_fit, xgb_val_processed) %>%
  bind_cols(truth = xgb_val_processed$SalePriceLog)

metrics(xgb_preds, truth=truth, estimate=.pred)

rmsle_preds_xgb <- expm1(xgb_preds$.pred)
rmsle_actual_xgb <- expm1(xgb_preds$truth)
rmsle(rmsle_preds_xgb, rmsle_actual_xgb)

library(vip)
vip(final_xgb_fit$fit$fit)

###################### Final Test
test_ids <- test$Id

lm_test_processed <- bake(linear_rec, new_data=test)
lm_test_preds <- predict(lm_fit, new_data=lm_test_processed) %>%
  mutate(SalePrice = expm1(.pred))

xgb_test_processed <- bake(xgb_rec, new_data=test)
xgb_test_preds <- predict(final_xgb_fit, new_data = xgb_test_processed) %>%
  mutate(SalePrice = expm1(.pred))  # undo the log transform

submission <- tibble(
  Id = test_ids,
  SalePrice_XGB = xgb_test_preds$SalePrice,
  SalePrice_LM = lm_test_preds$SalePrice
)

library(readr)
write_csv(submission, "outputs/submission.csv")




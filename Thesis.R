# Load required libraries
library(tidyverse)
library(caret)
library(e1071)         # For SVM
library(randomForest)
library(gbm)           # Gradient Boosting
library(xgboost)       # XGBoost

# Set seed using your student ID
set.seed(311441)

# Load the dataset
data <- read.csv("/Users/bentebeelen/documents/shopping_behavior_updated.csv")
summary(data)

# Convert target and relevant predictors to factors
data$Discount.Applied <- as.factor(data$Discount.Applied)
data$Promo.Code.Used <- as.factor(data$Promo.Code.Used)
data$Shipping.Type <- as.factor(data$Shipping.Type)

# Define the outcome and predictors
data_model <- data %>%
  select(Purchase.Amount..USD., Discount.Applied, Promo.Code.Used, Shipping.Type)

# Split the dataset
trainIndex <- createDataPartition(data_model$Purchase.Amount..USD., p = 0.6, list = FALSE)
train <- data_model[trainIndex, ]
temp <- data_model[-trainIndex, ]
validIndex <- createDataPartition(temp$Purchase.Amount..USD., p = 0.5, list = FALSE)
valid <- temp[validIndex, ]
test <- temp[-validIndex, ]

# ---- LINEAR REGRESSION ----
lm_model <- lm(Purchase.Amount..USD. ~ ., data = train)
lm_pred <- predict(lm_model, newdata = test)

# ---- SVM ----
svm_model <- svm(Purchase.Amount..USD. ~ ., data = train)
svm_pred <- predict(svm_model, newdata = test)

# ---- RANDOM FOREST ----
rf_model <- randomForest(Purchase.Amount..USD. ~ ., data = train)
rf_pred <- predict(rf_model, newdata = test)

# ---- GRADIENT BOOSTING ----
gbm_model <- gbm(Purchase.Amount..USD. ~ ., data = train, distribution = "gaussian", n.trees = 100)
gbm_pred <- predict(gbm_model, newdata = test, n.trees = 100)

# ---- XGBOOST ----
# Convert data for XGBoost
train_x <- model.matrix(Purchase.Amount..USD. ~ . -1, data = train)
train_y <- train$Purchase.Amount..USD.
test_x <- model.matrix(Purchase.Amount..USD. ~ . -1, data = test)

xgb_model <- xgboost(data = train_x, label = train_y, objective = "reg:squarederror", nrounds = 100, verbose = 0)
xgb_pred <- predict(xgb_model, newdata = test_x)

# ---- EVALUATION ----
evaluate <- function(pred, true) {
  pred <- as.numeric(pred)
  true <- as.numeric(true)
  
  RMSE <- sqrt(mean((pred - true)^2, na.rm = TRUE))
  MAE <- mean(abs(pred - true), na.rm = TRUE)
  
  # Create default MAPE in case all true values are zero
  MAPE <- NA
  
  if (any(true != 0)) {
    nonzero <- true != 0
    mape_vals <- abs((pred[nonzero] - true[nonzero]) / true[nonzero])
    # Check for any NaN or Inf
    mape_vals <- mape_vals[is.finite(mape_vals)]
    if (length(mape_vals) > 0) {
      MAPE <- mean(mape_vals) * 100
    }
  }
  
  return(data.frame(RMSE = RMSE, MAE = MAE, MAPE = MAPE))
}


results <- list(
  Linear_Regression = evaluate(lm_pred, test$Purchase.Amount..USD.),
  SVM = evaluate(svm_pred, test$Purchase.Amount..USD.),
  Random_Forest = evaluate(rf_pred, test$Purchase.Amount..USD.),
  Gradient_Boosting = evaluate(gbm_pred, test$Purchase.Amount..USD.),
  XGBoost = evaluate(xgb_pred, test$Purchase.Amount..USD.)
)

print(results)

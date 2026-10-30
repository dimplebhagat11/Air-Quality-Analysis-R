# ============================================================
# MINI PROJECT: Air Quality Analysis in New York (1973)
# Theme   : Environmental Science
# Project : EDA (Project 1) + Regression Analysis (Project 2)
# Dataset : airquality (built into R, no download needed)
# Goal    : Predict Ozone level using Temp, Wind and Solar.R
# ============================================================

# ------------------------------------------------------------
# 1. LOAD AND LOOK AT THE DATA
# ------------------------------------------------------------
data(airquality)
df <- airquality

head(df)          # first 6 rows
str(df)           # structure of data
summary(df)       # summary statistics

# Variables:
# Ozone   = ozone level (ppb)          <- dependent variable
# Solar.R = solar radiation (lang)     <- independent
# Wind    = average wind speed (mph)   <- independent
# Temp    = max daily temperature (F)  <- independent
# Month, Day = date

# ------------------------------------------------------------
# 2. DATA CLEANING (missing values)
# ------------------------------------------------------------
cat("\nMissing values in each column:\n")
print(colSums(is.na(df)))

df <- na.omit(df)   # remove rows with missing values
cat("\nRows after removing NA:", nrow(df), "\n")

# ------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------
vars <- c("Ozone", "Solar.R", "Wind", "Temp")

# 3a. Descriptive statistics using a loop
cat("\n--- Descriptive Statistics ---\n")
for (v in vars) {
  cat(v, ": mean =", round(mean(df[[v]]), 2),
      ", median =", median(df[[v]]),
      ", sd =", round(sd(df[[v]]), 2), "\n")
}

# 3b. Histograms (distribution of each variable)
par(mfrow = c(2, 2))
for (v in vars) {
  hist(df[[v]], main = paste("Histogram of", v),
       xlab = v, col = "skyblue", border = "white")
}

# 3c. Boxplots and outliers
par(mfrow = c(2, 2))
for (v in vars) {
  boxplot(df[[v]], main = paste("Boxplot of", v),
          col = "orange", horizontal = TRUE)
  out <- boxplot.stats(df[[v]])$out
  cat(v, "has", length(out), "outlier(s):", out, "\n")
}

# 3d. Boxplot of Ozone by Month
par(mfrow = c(1, 1))
boxplot(Ozone ~ Month, data = df,
        main = "Ozone Level by Month",
        xlab = "Month (5 = May ... 9 = September)",
        ylab = "Ozone (ppb)", col = "lightgreen")

# 3e. Scatter plots: Ozone vs each predictor
par(mfrow = c(1, 3))
predictors <- c("Solar.R", "Wind", "Temp")
for (p in predictors) {
  plot(df[[p]], df$Ozone,
       main = paste("Ozone vs", p),
       xlab = p, ylab = "Ozone", pch = 19, col = "blue")
  abline(lm(df$Ozone ~ df[[p]]), col = "red", lwd = 2)
}
par(mfrow = c(1, 1))

# 3f. Correlation
cat("\n--- Correlation Matrix ---\n")
print(round(cor(df[, vars]), 2))

# ------------------------------------------------------------
# 4. HYPOTHESIS TEST (t-test)
# H0: mean ozone is same in early summer (May-Jun) and
#     peak summer (Jul-Aug)
# ------------------------------------------------------------
early <- df$Ozone[df$Month %in% c(5, 6)]
peak  <- df$Ozone[df$Month %in% c(7, 8)]

cat("\n--- t-test: Early summer vs Peak summer ---\n")
print(t.test(early, peak))
# If p-value < 0.05, we reject H0 (ozone differs significantly)

# ------------------------------------------------------------
# 5. SPLIT DATA INTO TRAIN (70%) AND TEST (30%)
# ------------------------------------------------------------
set.seed(123)
train_index <- sample(1:nrow(df), 0.7 * nrow(df))
train <- df[train_index, ]
test  <- df[-train_index, ]

# ------------------------------------------------------------
# 6. REGRESSION MODELS
# ------------------------------------------------------------
# Model 1: Multiple linear regression
model1 <- lm(Ozone ~ Temp + Wind + Solar.R, data = train)

# Model 2: Polynomial regression (Temp squared)
model2 <- lm(Ozone ~ Temp + I(Temp^2) + Wind + Solar.R, data = train)

cat("\n--- Model 1: Linear Regression ---\n")
print(summary(model1))

cat("\n--- Model 2: Polynomial Regression ---\n")
print(summary(model2))

# ------------------------------------------------------------
# 7. MODEL EVALUATION (loop over both models)
# ------------------------------------------------------------
models <- list(Linear = model1, Polynomial = model2)

cat("\n--- Model Comparison ---\n")
for (name in names(models)) {
  m <- models[[name]]
  pred <- predict(m, newdata = test)
  rmse <- sqrt(mean((test$Ozone - pred)^2))
  r2_train <- summary(m)$r.squared
  adj_r2   <- summary(m)$adj.r.squared
  cat(name, "-> R2 =", round(r2_train, 3),
      "| Adj R2 =", round(adj_r2, 3),
      "| Test RMSE =", round(rmse, 2), "\n")
}

# ------------------------------------------------------------
# 8. RESIDUAL PLOTS (check model assumptions) for the polynomial model
# ------------------------------------------------------------
par(mfrow = c(2, 2))
plot(model2)
par(mfrow = c(1, 1))

# Actual vs Predicted plot on test data
pred2 <- predict(model2, newdata = test)
plot(test$Ozone, pred2,
     main = "Actual vs Predicted Ozone (Test Data)",
     xlab = "Actual", ylab = "Predicted", pch = 19, col = "darkgreen")
abline(0, 1, col = "red", lwd = 2)

# ------------------------------------------------------------
# 9. PREDICT A NEW DAY
# ------------------------------------------------------------
new_day <- data.frame(Temp = 85, Wind = 8, Solar.R = 200)
cat("\nPredicted Ozone for Temp=85, Wind=8, Solar.R=200:",
    round(predict(model2, new_day), 2), "ppb\n")

# ------------------------------------------------------------
# 10. CONCLUSION (write in your report)
# ------------------------------------------------------------
# - Ozone is right-skewed and has a few high outliers.
# - Temp has a strong positive correlation with Ozone;
#   Wind has a negative correlation (more wind -> less ozone).
# - Ozone is highest in July-August (t-test confirms the difference).
# - The polynomial model usually fits better than the linear one
#   (higher Adj R2, lower RMSE) because the Temp-Ozone
#   relationship is curved, not a straight line.
# - Higher temperature and low wind speed are the main drivers
#   of high ozone pollution.


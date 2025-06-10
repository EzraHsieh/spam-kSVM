# Mini-Project 2 Template: kSVM with Spambase Dataset
# This script follows the general procedure outlined in the Week 8 exercise.

# --------------------------------------------------------------------------
# 1. SETUP: LOAD LIBRARIES AND DATA
# --------------------------------------------------------------------------

# Remove all objects in the environment to start fresh
rm(list = ls())

# Ensure the kernelTools package is installed and loaded.
# install.packages("kernelTools") # Run this if you haven't installed it yet
library(kernelTools)

# The Spambase dataset is available from the UCI Machine Learning Repository.
# Download "spambase.data" and "spambase.names" to your working directory.
# URL: https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/

# Load the data
# The data file has no header, so we'll load it and add column names.
spam.raw <- read.csv("spambase.data", header = FALSE)

# Load and assign feature names from the .names file
# The names file has a specific format, so we'll extract them.
spam.names <- readLines("spambase.names")
# The attribute names start on line 34 in the .names file
col_names <- sapply(strsplit(spam.names[34:90], ":"), function(x) x[1])
# The last name is the outcome variable, "spam"
col_names <- c(col_names, "spam")

# Assign the names to the columns of our data frame
colnames(spam.raw) <- col_names

# --------------------------------------------------------------------------
# 2. DATA PREPARATION
# --------------------------------------------------------------------------

# Separate predictors (x) and outcome (y)
# The last column (58) is the outcome variable.
x <- spam.raw[, 1:57]
y <- spam.raw[, 58]

# The outcome 'y' is already coded as 1 for spam, 0 for not-spam.
# For kSVM, it's good practice to use a logical vector.
y <- (y == 1)

# Set a seed for reproducibility
set.seed(1331)

# Hold out a fraction of data for training and testing
# We'll use an 80/20 split.
train_frac <- 0.8
n <- NROW(x)
train_indices <- sample(1:n, floor(train_frac * n))

x_train <- x[train_indices, ]
y_train <- y[train_indices]

x_test <- x[-train_indices, ]
y_test <- y[-train_indices]


# --------------------------------------------------------------------------
# 3. MODEL TUNING (kernSVMBW)
# --------------------------------------------------------------------------
# As per the Week 8 exercise, we'll tune the model using kernSVMBW to find
# the best kernel and regularization parameters.
# We will test three different kernels: RBF, polynomial, and arc-cosine.

# --- Option 1: Tune with Radial Basis Function (RBF) Kernel ---
# The RBF kernel is a very common and powerful choice for SVMs.
# It has one parameter, sigma, which controls the width of the kernel.
cat("Tuning with RBF dot kernel...\n")
tune_rbf <- kernSVMBW(
  y = y_train, x = x_train,
  kernel = "rbf",
  kern.param = list(sigma = 10^(-3:1)),  # Grid for sigma parameter
  reg.param = 10^(-3:3),                # Grid for regularization parameter
  reps = 1,                             # Using 1 fold of CV for speed, increase for robustness
  verbose = FALSE,                       # Set to TRUE to see progress
  k = 5 # Number of folds for cross-validation
)

print("Best RBF Parameters:")
print(tune_rbf)
cat("\n\n")


# --- Option 2: Tune with Polynomial Kernel ---
# The polynomial kernel maps data into a higher-dimensional space.
# It has three parameters: degree, scale, and offset. We'll tune the degree.
cat("Tuning with Polynomial dot kernel...\n")
tune_poly <- kernSVMBW(
  y = y_train, x = x_train,
  kernel = "poly",
  kern.param = list(degree = 1:3),      # Grid for polynomial degree
  reg.param = 10^(-3:3),
  reps = 1,
  verbose = FALSE,
  k = 5 # Number of folds for cross-validation
)

print("Best Polynomial Parameters:")
print(tune_poly)
cat("\n\n")


# --- Option 3: Tune with Arc-Cosine Kernel ---
# The arc-cosine kernel is useful for high-dimensional data like this.
# It has one parameter, the order, which we will tune.
cat("Tuning with Arc-Cosine kernel...\n")
tune_arccos <- kernSVMBW(
  y = y_train, x = x_train,
  kernel = "arccos",
  kern.param = list(0:2),               # Grid for arccos order
  reg.param = 10^(-3:3),
  reps = 1,
  verbose = FALSE,
  k = 5 # Number of folds for cross-validation
)

print("Best Arc-Cosine Parameters:")
print(tune_arccos)
cat("\n\n")

# --- Option 4: Tune with a Linear Kernel ---
# The linear kernel is equivalent to a standard (non-kernelized) SVM.
# It serves as an excellent baseline. A linear kernel has no hyperparameters
# of its own, so we only need to tune the regularization parameter.
cat("Tuning with Linear  kernel...\n")
tune_linear <- kernSVMBW(
  y = y_train, x = x_train,
  kernel = "linear",
  kern.param = list(),                  # No kernel-specific parameters to tune
  reg.param = 10^(-3:3),                # Grid for regularization parameter
  reps = 1,
  verbose = FALSE,
  k = 5 # Number of folds for cross-validation
)

print("Best Linear Kernel Parameters:")
print(tune_linear)
cat("\n\n")


# --------------------------------------------------------------------------
# 4. FINAL MODEL FITTING AND EVALUATION
# --------------------------------------------------------------------------

# After reviewing the tuning results, select the best kernel and parameters.
# For this template, we'll proceed with the best parameters from the RBF tuning.
# In your project, you should justify your choice based on the CV accuracy.
best_kernel <- "rbf"
best_kern_param <- tune_rbf$best.kernpar
best_reg_param <- tune_rbf$best.regpar

# Fit the final SVM model on the training data
cat("Fitting final model with", best_kernel, "kernel...\n")
fit_final <- kernSVM(
  y = y_train, x = x_train,
  kernel = best_kernel,
  kern.param = best_kern_param,
  reg.param = best_reg_param
)

# --- In-Sample Evaluation ---
cat("\n--- In-Sample Results (on Training Data) ---\n")
print("In-sample confusion matrix:")
print(fit_final$confusion)
in_sample_accuracy <- sum(diag(fit_final$confusion)) / length(y_train)
cat("In-sample Accuracy:", round(in_sample_accuracy, 4), "\n")


# --- Out-of-Sample Evaluation (on Test Data) ---
cat("\n--- Out-of-Sample Results (on Test Data) ---\n")
# Use the predict method for out-of-sample performance
y_pred_test <- predict(fit_final, x = x_test, type = "class")


# Create the out-of-sample confusion matrix
oos_confusion <- table(Actual = y_test, Predicted = y_pred_test)

# Manually calculate the correct accuracy
# Note: We treat predictions of '0' as incorrect for this calculation.
true_negatives <- oos_confusion["FALSE", "-1"]
true_positives <- oos_confusion["TRUE", "1"]

correct_predictions <- true_negatives + true_positives
total_predictions <- sum(oos_confusion)

oos_accuracy <- correct_predictions / total_predictions

cat("Correct Out-of-Sample Accuracy:", round(oos_accuracy, 4), "\n")


# --- Compare Accuracies ---
cat("\n--- Accuracy Comparison ---\n")
cat("In-Sample Accuracy:              ", round(in_sample_accuracy, 4), "\n")
cat("Cross-Validated Accuracy (from BW):", round(tune_rbf$best.performance, 4), "\n")
cat("Out-of-Sample Accuracy:          ", round(oos_accuracy, 4), "\n\n")


# --------------------------------------------------------------------------
# 5. FEATURE INTERPRETATION
# --------------------------------------------------------------------------
# To understand which features the model is using, we can correlate the raw
# predictors with the raw prediction score (y-hat).

# Get the raw prediction scores on the training data
y_hat_raw <- fit_final$y.pred

# Calculate correlations
feature_correlations <- cor(x_train, y_hat_raw)

# View the most influential features
cat("--- Top 10 Features Most Positively Correlated with 'Spam' Classification ---\n")
print(sort(feature_correlations[,1], decreasing = TRUE)[1:10])

cat("\n--- Top 10 Features Most Negatively Correlated with 'Spam' Classification (i.e., indicative of 'Not Spam') ---\n")
print(sort(feature_correlations[,1], decreasing = FALSE)[1:10])



# --------------------------------------------------------------------------
# 1. Performance Metrics Table
# --------------------------------------------------------------------------
# This code calculates key performance metrics from your confusion matrix.

# Extract the counts of True/False Positives/Negatives
# This assumes the table is structured as table(Actual, Predicted)
true_negatives <- oos_confusion["FALSE", "-1"]
true_positives <- oos_confusion["TRUE", "1"]
false_positives <- oos_confusion["FALSE", "1"]
false_negatives <- oos_confusion["TRUE", "-1"]

# Calculate key metrics
accuracy <- (true_positives + true_negatives) / sum(oos_confusion)
precision <- true_positives / (true_positives + false_positives)
recall <- true_positives / (true_positives + false_negatives)
f1_score <- 2 * (precision * recall) / (precision + recall)
specificity <- true_negatives / (true_negatives + false_positives)

# Create a clean data frame for the report
performance_table <- data.frame(
  Metric = c("Accuracy", "Precision (PPV)", "Recall (Sensitivity)", "F1-Score", "Specificity (TNR)"),
  Value = round(c(accuracy, precision, recall, f1_score, specificity), 4)
)

cat("\n\n--- Performance Metrics Summary Table ---\n")
print(performance_table)

# --- Generating a Confusion Matrix and Plot with the 'caret' Package ---

# Install the package if you don't have it
# install.packages("caret")
# install.packages("e1071") # caret may require this package

library(caret)

# For caret to work correctly, the predicted and actual values must be
# factors with the same levels. We'll convert them.
# The `labels` argument makes the output more readable.
pred_factor <- factor(y_pred_test,
                      levels = c("-1", "1"),
                      labels = c("Not Spam", "Spam"))

actual_factor <- factor(y_test,
                        levels = c(FALSE, TRUE),
                        labels = c("Not Spam", "Spam"))

# Create the confusion matrix object
# This object contains the matrix, metrics, and more.
cm_caret <- confusionMatrix(data = pred_factor, reference = actual_factor)

# Print the detailed results (matrix and all statistics)
cat("--- Detailed Metrics from the caret Package ---\n")
print(cm_caret)

# Visualize the matrix using the fourfoldplot
# This plot is great for showing the proportions of correct/incorrect predictions.
cat("\nGenerating fourfold plot...\n")
png("fourfold_plot.png", width = 6, height = 6, units = "in", res = 300)
fourfoldplot(cm_caret$table, color = c("lightcoral", "lightgreen"),
             main = "Out-of-Sample Confusion Matrix")
dev.off()

# --- Generating a Custom Confusion Matrix Plot with 'ggplot2' ---

# Install the package if you don't have it
# install.packages("ggplot2")

library(ggplot2)

# First, create a data frame from your confusion matrix table
# We will use the factor versions of the variables created for the caret example
cm_df <- as.data.frame(table(Actual = actual_factor, Predicted = pred_factor))
colnames(cm_df) <- c("Actual", "Predicted", "Count")

# Create the plot
gg_cm_plot <- ggplot(data = cm_df, aes(x = Predicted, y = Actual, fill = (Actual == Predicted))) +
  geom_tile(color = "white", lwd = 1.5) + # Add tile borders
  geom_text(aes(label = Count), vjust = 0.5, size = 6, fontface = "bold") +
  scale_fill_manual(values = c("TRUE" = "#5cb85c", "FALSE" = "#d9534f"), # Green/Red colors
                    name = "Classification",
                    labels = c("Incorrect", "Correct")) +
  theme_minimal() +
  labs(title = "Out-of-Sample Confusion Matrix",
       x = "Predicted Class",
       y = "Actual Class") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10),
    legend.position = "bottom"
  )

# Print the plot to the RStudio plot pane
print(gg_cm_plot)

# Save the plot to a file
ggsave("ggplot_confusion_matrix.png", plot = gg_cm_plot, width = 6, height = 6, units = "in", dpi = 300)

cat("\nCustom ggplot confusion matrix saved as 'ggplot_confusion_matrix.png'\n")
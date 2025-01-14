#Sahil Hemant Deshmukh
#ALY6050 MOD- 5 & 6.
#MAY 18TH 2024

cat("\014") # clears console
rm(list = ls()) # clears global environment
try(dev.off(dev.list()["RStudioGD"]), silent = TRUE) # clears plots
try(p_unload(p_loaded(), character.only = TRUE), silent = TRUE) #clears packages
options(scipen = 100) # disables scientific notation for entire R session

# Load necessary libraries
library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)
library(broom)

data <- read.csv("all_rides_df_clean.csv")

#Defining the oversampling function
oversample_minority <- function(data, target) {
  majority <- data %>% filter(!!sym(target) == 0)
  minority <- data %>% filter(!!sym(target) == 1)
  oversampled_minority <- minority %>% sample_n(nrow(majority), replace = TRUE)
  oversampled_data <- bind_rows(majority, oversampled_minority)
  return(oversampled_data)
}

set.seed(123) 
bike_data <- data %>% sample_frac(0.05)
head(bike_data)

#Cleaning the data by handling NA values
clean_data <- bike_data %>% drop_na()  

#Detecting and removing outliers based on the IQR method
outlier_removed_data <- clean_data %>%
  mutate(across(where(is.numeric), ~ {
    q1 <- quantile(., 0.25, na.rm = TRUE)
    q3 <- quantile(., 0.75, na.rm = TRUE)
    iqr <- q3 - q1
    ifelse(. < (q1 - 1.5 * iqr) | . > (q3 + 1.5 * iqr), NA, .)
  })) %>%
  drop_na()  

bike_data <- outlier_removed_data %>%
  mutate(long_ride = ifelse(ride_length_minutes > 120, 1, 0))

set.seed(123)
bike_data$num_rides_past_month <- sample(1:30, nrow(bike_data), replace = TRUE)

#Converting long_ride to a factor for classification
bike_data$long_ride <- as.factor(bike_data$long_ride)

#Removing non-relevant identifier variables
bike_data <- bike_data %>%
  select(-ride_id, -start_station_id, -end_station_id, -started_at, -ended_at, -start_lat, -start_lng, -end_lat, -end_lng, -started_at_date, -started_at_time, -ended_at_date, -ended_at_time, -ride_length, -ride_length_category)

#both classes are present
if (sum(bike_data$long_ride == 1) == 0) {
  # Mock some data for the minority class
  minority_class_data <- bike_data %>%
    sample_n(10) %>%
    mutate(long_ride = as.factor(1))
  bike_data <- bind_rows(bike_data, minority_class_data)
}

#Standardizing numeric predictors
bike_data <- bike_data %>%
  mutate(across(c(ride_length_minutes, num_rides_past_month), scale))

set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(bike_data$long_ride, p = 0.7, list = FALSE)
train_data <- bike_data[trainIndex, ]
test_data <- bike_data[-trainIndex, ]

train_data_balanced <- oversample_minority(train_data, "long_ride")

table(train_data_balanced$long_ride)

#Checking for multicollinearity using VIF
vif_data_subset <- train_data_balanced %>%
  select(ride_length_minutes, num_rides_past_month, rideable_type, season, part_of_day) %>%
  sample_frac(0.01)  # Use 1% of the data for VIF calculation
library(car)
vif_model_subset <- lm(as.numeric(long_ride) ~ ride_length_minutes + num_rides_past_month + rideable_type + season + part_of_day, data = train_data_balanced %>% sample_frac(0.01))
vif(vif_model_subset)

#Decision Tree Model
decision_tree <- rpart(long_ride ~ ride_length_minutes + num_rides_past_month + rideable_type + season + part_of_day, 
                       data = train_data_balanced)
rpart.plot(decision_tree)

importance_dt <- as.data.frame(varImp(decision_tree))
importance_dt <- tibble::rownames_to_column(importance_dt, "Feature")
ggplot(importance_dt, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance for Decision Tree",
       x = "Feature",
       y = "Importance")

#Random forest model
set.seed(123)
train_data_sampled <- train_data_balanced %>% sample_frac(0.01)  


rf_model <- randomForest(long_ride ~ ride_length_minutes + num_rides_past_month + rideable_type + season + part_of_day, 
                         data = train_data_sampled, 
                         importance = TRUE,
                         ntree = 100,  
                         mtry = 2)  
#Predicting using the Random Forest model
rf_predictions <- predict(rf_model, test_data)
rf_accuracy <- mean(rf_predictions == test_data$long_ride)
print(paste("Random Forest Accuracy:", rf_accuracy))

#Confusion matrix
confusion_matrix_rf <- table(Predicted = rf_predictions, Actual = test_data$long_ride)
print(confusion_matrix_rf)

#Visualization: Bar plot of ride_length_minutes grouped by member_casual
bike_data %>%
  filter(ride_length_minutes <= 200) %>%
  ggplot(aes(x = ride_length_minutes, fill = member_casual)) +
  geom_histogram(position = "dodge", binwidth = 10) +  # Adjust binwidth as needed
  labs(title = "Distribution of Ride Lengths by Membership Type",
       x = "Ride Length (minutes)",
       y = "Count") +
  theme_minimal()

#Visualization: Box plot of ride_length_minutes by rideable_type, faceted by member_casual
bike_data %>%
  filter(ride_length_minutes <= 200) %>%
  ggplot(aes(x = rideable_type, y = ride_length_minutes, fill = rideable_type)) +
  geom_boxplot() +
  facet_wrap(~ member_casual) +
  labs(title = "Box Plot of Ride Lengths by Rideable Type and Membership",
       x = "Rideable Type",
       y = "Ride Length (minutes)") +
  theme_minimal()

#Visualization: Heat map of season vs part_of_day
bike_data %>%
  count(season, part_of_day) %>%
  ggplot(aes(x = season, y = part_of_day, fill = n)) +
  geom_tile() +
  labs(title = "Interaction Between Season and Part of Day",
       x = "Season",
       y = "Part of Day") +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal()

#Visualization: Histogram of ride_length_minutes faceted by part_of_day
bike_data %>%
  filter(ride_length_minutes <= 200) %>%
  ggplot(aes(x = ride_length_minutes)) +
  geom_histogram(binwidth = 10, fill = "skyblue", color = "black") +
  facet_wrap(~ part_of_day) +
  labs(title = "Histogram of Ride Lengths by Part of Day",
       x = "Ride Length (minutes)",
       y = "Frequency") +
  theme_minimal()

bike_data <- bike_data %>%
  mutate(across(c(ride_length_minutes, num_rides_past_month), scale))

# Split the data into training and testing sets
set.seed(123)  
trainIndex <- createDataPartition(bike_data$long_ride, p = 0.7, list = FALSE)
train_data <- bike_data[trainIndex, ]
test_data <- bike_data[-trainIndex, ]

# Train the SVM model
svm_model <- svm(long_ride ~ ride_length_minutes + num_rides_past_month + rideable_type + season + part_of_day, 
                 data = train_data, 
                 kernel = "linear", 
                 probability = TRUE)

# Predict using the SVM model
svm_predictions <- predict(svm_model, test_data, probability = TRUE)

# Calculate accuracy
svm_accuracy <- mean(svm_predictions == test_data$long_ride)
print(paste("SVM Accuracy:", svm_accuracy))

# Confusion matrix
confusion_matrix <- table(Predicted = svm_predictions, Actual = test_data$long_ride)
print(confusion_matrix)





# Chicago-BikeRide-Prediction

## Project Overview
This project analyzes bike ride data to predict whether a ride will last longer than 2 hours. The goal is to understand the factors influencing long rides and to provide actionable insights for bike-sharing companies to optimize their services.

Key objectives:
- Identify patterns in ride duration based on user behavior, rideable type, and time of day.
- Develop machine learning models to classify rides as long (over 2 hours) or not.
- Compare model performance to choose the most effective and interpretable solution.

---

## Dataset
- **Description**: A dataset containing bicycle ride records with features like ride duration, user type, season, and part of the day.
- **Key Features**:
  - `ride_length_minutes`: Duration of the ride (in minutes).
  - `rideable_type`: Type of bike (classic, docked, electric).
  - `member_casual`: Rider type (member or casual).
  - `season`: Season during which the ride occurred.
  - `part_of_day`: Time of day (morning, afternoon, evening, night).

---

## Methodology
1. **Data Preprocessing**:
   - Removed missing values and outliers using the IQR method.
   - Created a binary target variable (`long_ride`) for rides over 120 minutes.
   - Standardized numeric features and balanced classes through oversampling.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized data distributions, trends, and interactions using bar plots, histograms, and heatmaps.
   - Checked multicollinearity using VIF to ensure model robustness.

3. **Machine Learning Models**:
   - **Decision Tree**: Chosen for its interpretability and ease of use.
   - **Random Forest**: Tested for higher accuracy but exhibited overfitting.
   - **Support Vector Machine (SVM)**: Achieved the highest accuracy but lacked interpretability.

4. **Model Evaluation**:
   - Metrics: Accuracy, Precision, Recall, and Confusion Matrix.
   - Emphasis on interpretability and generalization to avoid overfitting.

---

## Technologies Used
- **R**: Data preprocessing, visualization, and model building
- **Libraries**:
  - `dplyr`, `tidyr` for data manipulation
  - `ggplot2` for visualization
  - `caret`, `rpart`, `randomForest`, `e1071` for machine learning

---

## Results
1. **Decision Tree**:
   - **Accuracy**: 99.77%
   - Interpretable results with a clear decision-making process.

2. **Random Forest**:
   - **Accuracy**: 99.77%
   - Evidence of overfitting due to high accuracy but poor generalizability.

3. **Support Vector Machine (SVM)**:
   - **Accuracy**: 99.99%
   - High accuracy but heavily biased towards the majority class, leading to poor interpretability.

### Conclusion:
- The **Decision Tree** model was chosen for its balance between accuracy and interpretability. 
- Insights gained from the model can help bike-sharing companies understand and cater to user behavior effectively.

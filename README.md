# MINI-PROJECT-

# Obesity Prediction Based on Eating Habits and Physical Activities

## Description
This project aims to predict the level of obesity based on two main factors: eating habits and physical activities. By analyzing data collected on individuals' daily food intake and exercise routines, the model will predict obesity levels using machine learning algorithms. The goal is to provide insights for healthier lifestyle choices and early detection of obesity risk factors.

## Features
- Predicts obesity levels based on user input (food intake and physical activity)
- Displays recommendations for maintaining a healthy lifestyle based on the predicted obesity level
- Implements a machine learning model to classify obesity levels into categories
- Visual representation of data trends and model performance

## Technologies Used
- **Python**: The primary programming language used for data processing, model building, and analysis.
- **Scikit-Learn**: For building the machine learning model.
- **Pandas**: For data manipulation and preprocessing.
- **Matplotlib/Seaborn**: For visualizing data and results.
- **Google Colab**: For prototyping and running the code interactively.

## Installation

To get a local copy up and running, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/obesity-prediction.git

## Usage
After setting up the project, you can use the following command to start the script and get predictions:

Run the main Python script:

bash
Copy code
python predict_obesity.py
The program will ask for input data such as food intake and physical activity level and return an obesity risk prediction.

You can also explore the data analysis and training of the machine learning model by opening the  Google Colab:




## Acknowledgments:

Special thanks to the dataset providers for offering publicly available data on eating habits and physical activities.
Inspiration from previous studies and research papers on obesity prediction models.
Thanks to the machine learning community for valuable resources and tutorials on building predictive models.


## code : 
```
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
```

```
# Load the dataset
df = pd.read_csv('/content/ObesityDataSet_raw_and_data_sinthetic.csv')  # Update 'obesity_data.csv' with your actual file path
print("Dataset Loaded Successfully")

# Display first 5 rows to understand the structure
print(df.head())
```

```
# Check for missing values
print("Missing values in each column:\n", df.isnull().sum())

# Dropping duplicates if any
df.drop_duplicates(inplace=True)
print("Data after removing duplicates:", df.shape)
```

```
label_encoder = LabelEncoder()
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
# 'Family_history_with_overweight' corrected to 'family_history_with_overweight'

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Checking data types after encoding
print(df.dtypes)

```

```
# Separating features and target variable
X = df.drop('NObeyesdad', axis=1)  # 'NObeyesdad' is the target column in this dataset
y = df['NObeyesdad']

# Applying Standard Scaler for feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

```

```
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
```

```
# Initialize Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)
print("Model training complete.")
```

```
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

```
# Plotting feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
plt.title('Feature Importance')
plt.show()
```

```
# Saving the trained model using joblib
import joblib
joblib.dump(model, 'obesity_predictor_model.pkl')
print("Model saved as 'obesity_predictor_model.pkl'")

```

```
# Loading the model
loaded_model = joblib.load('obesity_predictor_model.pkl')

# Making new predictions
sample_data = [0, 1, 22, 1, 2, 0, 1, 3, 2, 1, 0,0,0,0,0,0]  # Update with sample values as per your feature columns, the original data had 16 features
sample_data = np.array(sample_data).reshape(1, -1)
scaled_sample = scaler.transform(sample_data)
prediction = loaded_model.predict(scaled_sample)
print(f"Predicted Obesity Level: {prediction[0]}")
```

# Ouput : 
Dataset Loaded 
![image](https://github.com/user-attachments/assets/31d2a237-56a4-4dca-ae72-343c00c654a1)

![image](https://github.com/user-attachments/assets/9caa5b39-f3d5-4834-9f21-ca377edc3912)

## Accuracy Score : 

![image](https://github.com/user-attachments/assets/2ef64149-baad-440f-8a06-3905705c37c9)

## Confusion Matrix :
![image](https://github.com/user-attachments/assets/18716664-6a55-49b5-96ed-1d195d019715)

![image](https://github.com/user-attachments/assets/97bd91ec-8487-4cf0-b100-a1cd2d7298e8)

## Conclusion

This mini project successfully demonstrates the potential of machine learning to predict obesity levels based on eating habits and physical activities. By using a dataset that includes information about individuals' daily food intake and exercise routines, we have built a predictive model that can classify obesity levels into different categories. 

Through the use of various machine learning techniques, including data preprocessing, feature selection, and model evaluation, the project has highlighted the importance of lifestyle factors in determining obesity risk. The results show a promising accuracy in predicting obesity levels, which can potentially be used for preventive health measures and personalized lifestyle recommendations.

This project also offers a foundation for future improvements, such as the integration of additional factors (e.g., sleep patterns, mental health) and the use of more advanced models to enhance prediction accuracy. In addition, the deployment of this model as a web or mobile application could provide real-time obesity risk assessments to users, making it a practical tool for health management.

Overall, this project not only demonstrates the application of machine learning in health but also encourages further exploration in the field of predictive health analytics.


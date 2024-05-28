# %%
"""
### AIN422 Assignment-1 Kazım Halil KESMÜK 
"""

# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
"""
### Reading Data and Some Preprocessing Operations
"""

# %%
df = pd.read_csv("medical_heart.csv")
df.head()

# %%
# It's clear there are only integer or float values for each feature  

# %%
df.describe()

# %%
# There is some mathematical information about features but controlling outliers will be done for each feature in next steps

# %%
df.isnull().sum()

# %%
# There is no missing value in this dataset
# Only continuous features will be selected  

# %%
continuous_features = ["age", "trtbps", "chol" ,"thalachh", "oldpeak"]

# %%
def plot_outliers(df, continuous_features):
    num_features = len(continuous_features)
    
    fig, axes = plt.subplots(nrows=(num_features + 1) // 2, ncols=2, figsize=(12, 4 * ((num_features + 1) // 2)))
    axes = axes.flatten()
    
    for i, column in enumerate(continuous_features):
        if column in df.columns:
            sns.boxplot(x=df[column], ax=axes[i])
            axes[i].set_title(f"Box Plot - {column}")
            axes[i].set_xlabel(column)
            
    plt.tight_layout()
    plt.show()
plot_outliers(df,continuous_features)  # age, trtbps, thalachh,  oldpeak

# %%
def checkOutlier(df,continuous_features):
    outliers = {}
    for column in continuous_features:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        if not column_outliers.empty:
            outliers[column] = column_outliers
    
    if outliers:
        for column, outlier_rows in outliers.items():
            print(f"Outliers in column '{column}':")
            print(outlier_rows)
            print()
    else:
        print("No outliers found in the DataFrame.")

    return outliers
    
outliers = checkOutlier(df,continuous_features)

# %%
# It looks like there are some outliers in our dataset. Since the number of these outliers is not very large and our data is not large enough, 
# these outliers will not be deleted. However, after the model is completed, if the performance of the model is low, a new model can be created without outliers.

# %%
"""
# Exploratory Data Analysis (EDA)
"""

# %%
"""
### Let's start with categorical (discreate) features
"""

# %%
categorical_list = ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall", "output"]

# %%
df_categoric = df.loc[:, categorical_list]
for i in categorical_list:
    plt.figure()
    sns.countplot(x = i, data = df_categoric, hue = "output")
    plt.title(i)

# %%
"""
### Understanding These Drawn Bar Plots
- If sex=0, the rate of experiencing a heart attack is higher.

- If cp=0, the rate of having a heart attack is low, but for other cp values, the number of people who have had a heart attack is higher than those who haven't.

- If restecg=1, the number of people who have had a heart attack is higher, but for other restecg values, the number of people who have had a heart attack is lower than those who haven't.

- If exng=0, the number of people who have had a heart attack is almost twice the number of those who haven't, but if exng=1, the number of people who haven't had a heart attack is approximately three times the number of those who have.

- It is observed that people with slp=2 have a high rate of experiencing a heart attack.

- Unlike other values of caa, when caa=0, the rate of experiencing a heart attack is quite high.

- It is clear that people with thall=2 have a high rate of experiencing a heart attack.
"""

# %%
"""
### Let's continue with numeric features
"""

# %%
numeric_list = ["age", "trtbps", "chol", "thalachh", "oldpeak", "output"]

# %%
df_numeric = df.loc[:, numeric_list]
sns.pairplot(df_numeric, hue = "output", diag_kind = "kde")
plt.show()

# %%
"""
### Understanding These Drawn Scatter Plots
- Middle-aged individuals have a higher probability of experiencing a heart attack.
- People with high chol values have a higher probability of experiencing a heart attack.
- People with high thalachh values have a higher probability of experiencing a heart attack.
"""

# %%
plt.figure(figsize = (14,10))
sns.heatmap(df.corr(), annot = True, fmt = ".1f", linewidths = .7)
plt.show()

# %%
"""
### Understanding the Correlation Matrix
In the previous section, the relationship between features and output was examined. If we comment on this matrix for the last time, 
it seems clear that the "fbs" column does not have any effect on the output. Therefore, this column can be dropped.
"""

# %%
df = df.drop(['fbs'], axis=1)
df.info()

# %%
"""
### Standardization And Creating X and Y for Train and Test
"""

# %%
"""
In this part, the dataframe will undergo standardization, and then it will be split into X and Y. The "output" column will form Y, while the remaining columns will form X. In this assignment, two different models will be tried (Logistic Regression and Random Forest). Therefore, in this part, the X and Y split will be performed in a way that both models can use.
"""

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler

# %%
df1 = df.copy()

# %%
categorical_list.remove("fbs")
# The fbs column was deleted from df, so this column must be deleted from categorical_list

# %%
df1 = pd.get_dummies(df1, columns = categorical_list[:-1], drop_first = True)
df1[df1.columns[-len(categorical_list[:-1]):]] = df1[df1.columns[-len(categorical_list[:-1]):]].astype(int)
df1.head()

# %%
X = df1.drop(["output"], axis = 1)
y = df1[["output"]]

# %%
X.head()

# %%
y.head()

# %%
X[numeric_list[:-1]] = scaler.fit_transform(X[numeric_list[:-1]])

# %%
X.head()

# %%
"""
# MODELLING
"""

# %%
"""
## Two Models for This Task
In the heart attack prediction model, two different machine learning algorithms were used: Logistic Regression and Random Forest. These algorithms are commonly preferred in classification problems and offer different approaches. The aim of the study is to find the most suitable model by comparing the performance of these two algorithms.

- ### Logistic Regression
Logistic Regression is a statistical machine learning algorithm that is frequently used in binary classification problems. This algorithm predicts class probabilities based on given inputs by modeling the relationship between independent variables and the dependent variable. In the heart attack prediction model, the patient's risk factors and clinical data were taken as inputs, and the probability of a heart attack was obtained as the output. One of the advantages of Logistic Regression is that the results are easily interpretable. The coefficients of the model show the effect of each independent variable on the probability of a heart attack. Additionally, Logistic Regression can help simplify the model and prevent overfitting by performing feature selection.

- ### Random Forest
Random Forest, on the other hand, is an ensemble learning method that combines multiple decision trees. Each decision tree is trained on a random subset of the data, and the final prediction is obtained by combining the predictions of the trees. Random Forest can capture complex relationships between variables and handle high-dimensional datasets. One of the advantages of this algorithm is that it reduces the risk of overfitting. By using multiple decision trees, the generalization ability of the model is increased. Moreover, by calculating feature importances, it can be determined which risk factors are more important for heart attack prediction.


In the development of the heart attack prediction model, the different approaches and advantages of Logistic Regression and Random Forest algorithms were considered. The performance of the model was thoroughly evaluated using these two algorithms.
"""

# %%
"""
## Evaluation Metrics 



- ### Recall:
    Recall measures how well the model identifies the actual positive cases. In the heart attack prediction model, recall shows how many people who actually had a heart attack were correctly predicted by the model. It aims to minimize false negatives.
    
    
    The main reason for using recall in this model is that false negatives can have serious consequences. If the model incorrectly classifies a person who actually had a heart attack as healthy (false negative), it can prevent the patient from getting necessary treatment and potentially put their life at risk. Therefore, it's crucial for the model to detect heart attacks with high accuracy and minimize false negatives.
    
    
    Recall helps us evaluate the model's performance in this regard. A high recall value indicates that the model successfully captures the actual heart attack cases, helping patients receive early diagnosis and treatment.


- ### F1 Score:
    The F1 score is the harmonic mean of precision and recall. Precision measures how many of the positively classified cases are actually positive, while recall measures how many of the actual positives are correctly predicted. The F1 score combines these two metrics into a single value and provides a balanced evaluation of the model's overall performance.
    
    
    Using the F1 score in the heart attack prediction model ensures that the model balances both false positives (classifying healthy individuals as having a heart attack) and false negatives (classifying individuals who actually had a heart attack as healthy). A high F1 score indicates that the model achieves a good balance between precision and recall.
    
    
    The F1 score summarizes the model's overall performance in a single value, making it useful for comparing different models and selecting the most suitable one. It also provides a robust evaluation when the class distribution in the dataset is imbalanced.


In summary, using recall and F1 score metrics in the heart attack prediction model allows for a comprehensive evaluation of the model's performance. Recall focuses on minimizing false negatives, ensuring early diagnosis and treatment for patients, while the F1 score provides a balanced assessment of the model's overall performance. Using these metrics plays a critical role in evaluating the effectiveness and reliability of the heart attack prediction model.
"""

# %%
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, confusion_matrix, f1_score

def model_selection_and_evaluation(X, y, model_choice, test_size):
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Model selection and hyperparameter optimization
    if model_choice == "Logistic Regression":
        model = LogisticRegression()
        param_dist = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    else:
        raise ValueError("Unvalid Model Selection !")
    
    # Hyperparameter optimization
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=5, random_state=42)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    # Modelling and predicting
    best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    
    recall = recall_score(y_test, y_pred)
    
    f1 = f1_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)

    # Results
    print(f"Choosen Model: {model_choice}")
    print(f"Best Hyperparameters: {random_search.best_params_}")
    print(f"Test size : {test_size}")
    print(f"Test Set Recall Score: {recall}")
    print(f"Test Set F1 Score: {f1}")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, np.unique(y_test), rotation=45)
    plt.yticks(tick_marks, np.unique(y_test))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()


# %%
model_selection_and_evaluation(X,y,"Logistic Regression", 0.2)

# %%
model_selection_and_evaluation(X,y,"Random Forest", 0.2)

# %%
"""
For this project, two different classification models were tried: Logistic Regression and Random Forest. Our goal was to find the best performing model for the given dataset.


First, a Logistic Regression model was trained with specific hyperparameters. The 'solver' parameter was set to 'saga', which is suitable for L1 regularization. The 'penalty' parameter was set to 'l1', indicating the use of L1 regularization, which can help with feature selection. The 'C' parameter, which controls the inverse of the regularization strength, was set to 10. Using a test set size of 0.2, the model achieved a recall score of 0.875 and an F1 score of 0.889.


Next, a Random Forest model was trained with its own set of hyperparameters. The 'n_estimators' parameter, which determines the number of trees in the forest, was set to 200. The 'min_samples_split' parameter, which controls the minimum number of samples required to split a node, was set to 5. The 'min_samples_leaf' parameter, which determines the minimum number of samples required in a leaf node, was set to 1. The 'max_depth' parameter, which limits the maximum depth of the trees, was set to 5, restricting the depth of the trees. With the same test set size of 0.2, the model achieved a recall score of 0.813 and an F1 score of 0.852.


When the two models were compared, the Logistic Regression model performed slightly better in terms of both recall and F1 scores. However, the difference in performance was not significant, and both models demonstrated satisfactory results.


Overall, both models showed promising results, but I would still prefer the Logistic Regression model. Although the difference is small, Logistic Regression achieved better scores. The slight advantage in recall and F1 scores indicates that Logistic Regression performed a bit better for this dataset. Of course, this is just my opinion. Since the results are very close, some might prefer Random Forest. But for me, Logistic Regression gave slightly better results, so I would choose that one.


To put it simply, without getting too technical: there is no big difference between the two models, but Logistic Regression performed slightly better, so I would prefer it. However, since the results are very close, other factors should also be considered. But overall, Logistic Regression seems to be a bit more suitable for this dataset.
"""
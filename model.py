import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore') 


# Load the dataset
df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
print("Shape of the dataset:", df.shape)
print("/nFirst 5 rows of the dataset:")
print(df.head())
print("/nColumn names:")
print(df.columns.tolist())
print("/nMissing values:")
print(df.isnull().sum()) 


df = df.drop(['EmployeeCount', 'EmployeeNumber', 
               'Over18', 'StandardHours'], axis=1)


le = LabelEncoder()
text_columns = df.select_dtypes(include=['object']).columns.to_list()
print("Text columns:", text_columns)

for col in text_columns:
    df[col] = le.fit_transform(df[col])
    
print("after encoding:") 
print(df.head()) 



X = df.drop('Attrition', axis=1)
y = df['Attrition'] 
print("features shape:", X.shape)
print("target shape:", y.shape)
print("/nTarget value counts:")
print(y.value_counts()) 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training size:", X_train.shape)
print("Testing size:", X_test.shape) 


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")



models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Naive Bayes': GaussianNB()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}") 
    



best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

joblib.dump(best_model, 'model.pkl')
print(f"Best model saved as model.pkl")


y_pred_best = best_model.predict(X_test)

print("/nClassification Report:")
print(classification_report(y_test, y_pred_best))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'left'], yticklabels=['Stayed', 'left'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion matrix saved as confusion_matrix.png")


if best_model_name == 'Random Forest':
    feature_names = df.drop('Attrition', axis=1).columns
    importances = best_model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False).head(10)
    
    plt.figure(figsize=(8,5))
    sns.barplot(x='Importance', y='Feature', data=feat_df,palette='viridis')
    plt.title('Top 10 Feature Importances - Random Forest')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    print("Feature importance saved as feature_importance.png")
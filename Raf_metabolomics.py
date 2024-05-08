import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from a CSV file
file_path = 'path_to_your_file.csv'
df = pd.read_csv(file_path)

# Basic data check
print(df.head())

# Fit a mixed-effects model for each metabolite
# Let's assume your time column is named 'Time', the treatment column is 'Treatment', and the subject ID column is 'SubjectID'
results = {}
for metabolite in df.columns[3:]:  # Adjust the index to match where metabolite columns start
    formula = f"{metabolite} ~ Time + Treatment + (1|SubjectID)"
    model = smf.mixedlm(formula, df, groups=df['SubjectID'])
    result = model.fit()
    results[metabolite] = result
    print(f"Results for {metabolite}:")
    print(result.summary())

# Choose one metabolite's residuals for machine learning demonstration
df['Residuals'] = results['Metabolite_1'].resid  # Replace 'Metabolite_1' with the actual column name

# Prepare data for Random Forest Classifier
X = df[['Residuals']]  # Consider using more features or different feature engineering
y = df['Treatment']   # Assuming binary treatment variable: 0 = Placebo, 1 = Prebiotics

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions and evaluation
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of the classifier: {accuracy:.2f}")

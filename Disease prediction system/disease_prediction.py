import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


train_df = pd.read_csv("Training.csv")
test_df  = pd.read_csv("Testing.csv")


train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]


le = LabelEncoder()
train_df["prognosis"] = le.fit_transform(train_df["prognosis"])
test_df["prognosis"]  = le.transform(test_df["prognosis"])

X_train = train_df.drop("prognosis", axis=1)
y_train = train_df["prognosis"]
X_test  = test_df.drop("prognosis", axis=1)
y_test  = test_df["prognosis"]


models = {
    "Random Forest":   RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree":   DecisionTreeClassifier(random_state=42),
    "Naive Bayes":     GaussianNB()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    results[name] = {"model": model, "accuracy": acc}
    print(f"{name}: {acc*100:.2f}%")


best_name  = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_name]["model"]
print(f"\n✅ Best Model: {best_name}")


cm    = confusion_matrix(y_test, preds)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix – {best_name}")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig("confusion_matrix.png"); plt.show()


plt.figure(figsize=(7, 4))
names = list(results.keys())
accs  = [results[n]["accuracy"]*100 for n in names]
sns.barplot(x=names, y=accs, palette="viridis")
plt.ylim(90, 101)
plt.ylabel("Accuracy (%)"); plt.title("Model Accuracy Comparison")
plt.tight_layout(); plt.savefig("accuracy_comparison.png"); plt.show()


def predict_disease(symptoms_list):
    """symptoms_list: list of symptom column names that are present"""
    input_data = pd.DataFrame([np.zeros(len(X_train.columns))],
                               columns=X_train.columns)
    for symptom in symptoms_list:
        if symptom in input_data.columns:
            input_data[symptom] = 1
    prediction = best_model.predict(input_data)
    disease    = le.inverse_transform(prediction)
    return disease[0]

sample_symptoms = ["high_fever", "headache", "nausea", "constipation", "abdominal_pain"]
result = predict_disease(sample_symptoms)
print(f"\n🩺 Predicted Disease: {result}")
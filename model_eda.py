import pandas as pd
from df import monitoring, clean_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn


df_clean = clean_data(monitoring())[["Body_clean", "Tag"]]

# Divisez les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df_clean['Body_clean'], df_clean['Tag'], test_size=0.2, random_state=42, stratify=df_clean["Tag"])

# Vectorisation du texte
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Choisissez un modèle de classification (Naive Bayes ici)
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Évaluation du modèle
y_pred = clf.predict(X_test_vectorized)
print(classification_report(y_test, y_pred))


experiment_name = "projet_nlp_tag"
run_name = "baseline"

mlflow.set_experiment(experiment_name)

runs = mlflow.search_runs(experiment_names=["projet_nlp_tag"])

with mlflow.start_run(run_name=run_name):
    # Enregistrez les paramètres du modèle
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_param("classifier", "MultinomialNB")

    # Enregistrez les métriques d'évaluation
    report = classification_report(y_test, y_pred, output_dict=True)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", metric_value)
        else:
            mlflow.log_metric(label, metrics)



    # Enregistrez le modèle dans le format MLflow
    mlflow.sklearn.log_model(clf, "model")



def model():
    return clf

def vect():
    return vectorizer

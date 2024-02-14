import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import streamlit as st

# Step 1: Data Collection
#@st.cache
def load_data():
    data = pd.read_csv('onlinefraud.csv')
    return data

def main():
    st.title('E-commerce Fraud Detection')

    # Load the dataset
    data = load_data()

    # Display dataset
    st.subheader('Dataset')
    st.write(data.head())

    # Check for null values
    st.subheader('Null Values')
    st.write(data.isnull().sum())

    # Exploring transaction type
    st.subheader('Distribution of Transaction Type')
    type_counts = data['type'].value_counts()
    transactions = type_counts.index
    quantity = type_counts.values

    fig = go.Figure(data=go.Pie(labels=transactions, values=quantity, hole=0.5))
    fig.update_layout(title="Distribution of Transaction Type")
    st.plotly_chart(fig)

    # Checking correlation
    st.subheader('Correlation with "isFraud"')
    correlation = data.corr()['isFraud'].sort_values(ascending=False)
    st.write(correlation)

    # Transform categorical features
    data['type'] = data['type'].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
    data['isFraud'] = data['isFraud'].map({0: "No Fraud", 1: "Fraud"})

    # Splitting the data
    x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
    y = np.array(data[["isFraud"]])

    # Training a machine learning model
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    accuracy_values = []

    for model_name, model in models.items():
        model.fit(xtrain, ytrain)
        accuracy = model.score(xtest, ytest)

        # Display model accuracy
        st.subheader(f'{model_name} Accuracy')
        st.write(accuracy)

        # Store accuracy value
        accuracy_values.append(accuracy)

        # Make prediction
        sample_features = np.array([[1, 8900.2, 8990.2, 0.0]])
        prediction = model.predict(sample_features)

        # Display prediction
        st.subheader(f'{model_name} Prediction')
        st.write(prediction)

        # Predict labels for the test set
        y_pred = model.predict(xtest)

        # Evaluate the model's performance
        st.subheader(f'{model_name} Classification Report')
        st.write(classification_report(ytest, y_pred))

        st.subheader(f'{model_name} Confusion Matrix')
        st.write(confusion_matrix(ytest, y_pred))

    # Plot accuracy values
    fig = go.Figure(data=go.Bar(x=list(models.keys()), y=accuracy_values))
    fig.update_layout(title="Accuracy for Different Models", xaxis_title="Models", yaxis_title="Accuracy")
    st.plotly_chart(fig)

    # Compute the distribution of transaction types
    type_counts = data['type'].value_counts()
    transactions = type_counts.index
    quantity = type_counts.values

    # Create a pie chart
    fig = go.Figure(data=go.Pie(labels=transactions, values=quantity, hole=0.5))
    fig.update_layout(title="Distribution of Transaction Type")

    # Display the chart
    st.subheader('Distribution of Transaction Type (Chart)')
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()

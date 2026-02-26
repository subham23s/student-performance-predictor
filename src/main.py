import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    # Load dataset
    data = pd.read_csv("../data/student_data.csv")

    # Features and target
    X = data[["hours_studied", "attendance", "previous_score"]]
    y = data["final_score"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, predictions)

    print("Model trained successfully.")
    print("Mean Squared Error:", mse)

if __name__ == "__main__":
    main()

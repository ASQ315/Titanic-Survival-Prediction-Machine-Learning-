🚢 Titanic Survival Prediction (Machine Learning)

This project uses Logistic Regression to predict passenger survival on the Titanic dataset.
It demonstrates data preprocessing, model training, prediction generation, and visualization using Python.

📁 Project Files
```
- titanic_ML.py            # Main machine learning script
- titaniic_train.csv       # Training dataset
- titanic_test.csv         # Test dataset
- titani_prediction.csv    # Output predictions
```
🧠 Machine Learning Model
- **Algorithm**: Logistic Regression
- **Type**: Supervised Binary Classification
- **Target Variable**: `Survived` (0 = No, 1 = Yes)

🧾 Features Used

The model is trained using the following features:
- `Pclass` - Passenger class (1st, 2nd, 3rd)
- `Age` - Passenger age
- `SibSp` - Number of siblings/spouses aboard

🛠️ Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn

⚙️ How it Works
1. Reads the training dataset
2. Cleans missing values using median imputation
3. Trains a **Logistic Regression** model
4. Reads the **test dataset**
5. Predicts survival outcomes
6. Saves predictions to a CSV file
7. Visualizes training vs predicted data

▶️ How to Run the Project

1. Install dependecies
   ```
   pip install pandas matplotlib scikit-learn
   ```
2. Run the script
   ```
    python titanic_ML.py
   ```
📊 Output
- A file cnamed `titanic_predictions.csv` will be created
- Two visualization will be displayed:
  - Training data survival distribution
  - Predicted survival on test data

📈 Visualization
- Scatter plot of **Age vs Passenger Class**
- Color-coded survival outcomes:
  - '0' = Did not survive
  - '1' = Survived

📌 Example Prediction Output
```
Pclass,Age,SibSp,Survived_Predicted
3,22,1,0
1,38,1,1
```


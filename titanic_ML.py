#Use to supress warning messages in Python
import warnings; warnings.filterwarnings('ignore')
#Use to import Pandas library use for data manipulation and analysis
import pandas as pd 
from pandas import DataFrame
#Use to import pyplot module from Matplotlib library 
import matplotlib.pyplot as plt
#Imports the LogisticRegression class from scikit-learn library'ss linear_model module
#LogisticRegression is used for binary classification task 
from sklearn.linear_model import LogisticRegression

#This function CSV_reader_train is designed to read a CSV file and prepare the training data 
def CSV_reader_train(filename: str) -> DataFrame:
    #Open the CSV file in read mode
    with open(filename, 'r') as f: 
        #Read the CSV file into a Pandas DataFrame 
        df = pd.read_csv(f) 
        #Selects the specifics colums and drop the rows with missing values
        df_train_sel = df[['Survived','Pclass', 'Age', 'SibSp']].dropna()
        #Call the training_data function to process the selected areas from df_train_sel
        return training_data(df_train_sel)

#This function process the training data and trains a LogisticRegression model
def training_data(df_train_sel):
    
    #Fill the missing values in 'Pclass' column with the median value
    df_train_sel['Pclass'].fillna(df_train_sel['Pclass'].median(), inplace = True)
    
    #Fill the missing values in 'Age' column with the median value
    df_train_sel['Age'].fillna(df_train_sel['Age'].median(), inplace = True)
    
    #Fill the missing values in 'SibSp' column with the median value
    df_train_sel['SibSp'].fillna(df_train_sel['SibSp'].median(), inplace = True)

    #Select features (independent variables) for training
    titanic_X_train = df_train_sel[['Pclass', 'Age', 'SibSp']] 
    #Select target (dependent variables) for training
    titanic_Y_train = df_train_sel['Survived']
    
    #Initialize the Logistic Regression model
    model = LogisticRegression()
    #Train the model usiwng the training data
    model.fit(titanic_X_train, titanic_Y_train)
    
    #Return the training model and scaler for later use
    return model, titanic_X_train, titanic_Y_train

#This function reads a CSV file and prepares the test data
def CSV_reader_test(filename: str, model) -> DataFrame:
    #Open the CSV file in read mode
    with open(filename, 'r') as f:
        #Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(f)
        #Select specific colums and drop rows with missing values
        df_test_sel = df[['Pclass', 'Age', 'SibSp']].dropna()
        #Call the test_data function to process the selected data
        return test_data(df_test_sel, model)

#This function processes the test data and makes predictions using the trained mode
def test_data(df_test_sel, model):

    #Fill missing values in 'Pclass' column with the median value
    df_test_sel['Pclass'].fillna(df_test_sel['Pclass'].median(), inplace = True)
    
    #Fill missing values in 'Age' column with the median value
    df_test_sel['Age'].fillna(df_test_sel['Age'].median(), inplace = True)
    
    #Fill missing values in 'SibSp' column with the median value
    df_test_sel['SibSp'].fillna(df_test_sel['SibSp'].median(), inplace = True)

    #Select features (independenet variables) for testming
    titanic_X_test = df_test_sel[['Pclass', 'Age', 'SibSp']] 

    #Make predictions using the training moddel and add them to the DataFrame
    df_test_sel['Survived_Predicted'] = model.predict(titanic_X_test)
    
    #Save the predictions to a CSV file
    df_test_sel.to_csv("titanic_predictions.csv", index = False)
    print("Predictions saved to 'titanic_predictions.csv'")
    
    #Call the function to visualize the test data
    plot_visualization(titanic_X_test, df_test_sel, titanic_X_train, titanic_Y_train)
    
    #Return the DataFrame with predictions
    return df_test_sel

#This function visualizing the test data and predictions
def plot_visualization(titanic_X_test, df_test_sel, titanic_X_train, titanic_Y_train):
    
    #Create a new figure with the specific sizes
    plt.figure(figsize = (13, 5))
    
    #~~Plot for the training data~~
    plt.subplot(1, 2, 1)
     #Create a scatter plot of 'Age' vs 'Pclass', colored by survival status
    #Cmap is the color of the map plot visualization
    plt.scatter(titanic_X_train['Age'], titanic_X_train['Pclass'], c = titanic_Y_train, cmap = 'coolwarm')
    
    #Label for the X-asis
    plt.xlabel('Age')   
    
    #Define the categories for the Y-axis
    categories = [1, 2, 3]
    #Label for the Y-axis
    plt.ylabel('Pclass')
    #Set the Y-axis ticks to the defined categories
    plt.yticks(categories)
    
    #Set the title of the plot
    plt.title('Survival Training Data (No = 0, Yes = 1)')
    # Add a color bar to indicate survival status
    plt.colorbar(label = "Survival")
    
    #~~Plot for the testing data~~
    
    plt.subplot(1, 2, 2)
    #Create a scatter plot of 'Age' vs 'Pclass', colored by the predicted survival status
    plt.scatter(titanic_X_test['Age'], titanic_X_test['Pclass'], c = df_test_sel['Survived_Predicted'], cmap = 'coolwarm')
    
    #Label for the X-asis
    plt.xlabel('Age')
    
    #Define categories for the Y-axis
    categories = [1, 2, 3]
    #Label the Y-axis
    plt.ylabel('Pclass')
    #Set Y-axis ticks to the defined catagories
    plt.yticks(categories)
    
    #Set the title of the plot
    plt.title('Predicted Survival (No = 0, Yes = 1)')
    # Add a color bar to indicate predicted survival status
    plt.colorbar(label = "Predicted Survival Test")
    #Display the plot
    plt.show()

#Read the training data, train the model, and get the model and scaler
model , titanic_X_train, titanic_Y_train = CSV_reader_train('./titaniic_train.csv')

#Read the test data, make predictions and visualize the results
titanic_test_df = CSV_reader_test('./titanic_test.csv', model)
    


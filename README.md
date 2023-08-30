# Waiter_Tips_Prediction_with_Machine_Learning
Waiter Tips Prediction with Machine Learning.

In my waiter tips prediction project with machine learning, I utilized various data analysis and machine learning libraries and frameworks to create a predictive model.

First, I imported the necessary libraries such as pandas, numpy, and Plotly to handle the data and visualize the information.

Next, I loaded the data from the "tips.csv" CSV file using pandas and examined the first rows of the dataset to understand its structure.

To explore the data, I created interactive visualizations using Plotly Express and Plotly Graph Objects. I generated scatter plots that related the total bill amount ("total_bill") to the tip amount ("tip"), highlighting information such as table size, day of the week, customer gender, and mealtime. I also added trendlines using linear regression.

Additionally, I created pie charts to analyze the distribution of tips based on the day of the week, customer gender, smoking status, and mealtime.

Now, regarding the tip prediction model:

I performed data preprocessing, converting categorical variables into numerical ones. I mapped "Female" to 0 and "Male" to 1 in the "sex" attribute, "No" to 0 and "Yes" to 1 in "smoker," and so on.

I separated the independent variables (features) and the dependent variable (the tip) from the dataset.

I split the data into training and testing sets using scikit-learn's train_test_split, with an 80% training and 20% testing ratio.

I used a scikit-learn linear regression model to train the prediction model with the training data.

Finally, to test the model, I provided a set of example features and used the trained model to predict the expected tip based on these features.

This project allowed me to explore and visualize data related to restaurant tips and create a model that can predict tips based on various factors, including the bill amount, customer gender, smoking status, day of the week, mealtime, and table size.

# Waiter Tips Prediction with Machine Learning

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv("tips.csv")
print(data.head())

figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "day", trendline="ols")
figure.show()

figure = px.scatter(data_frame = data, x="total_bill",
                     y="tip", size="size", color= "sex", trendline="ols")
figure.show()

figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "time", trendline="ols")
figure.show()

figure = px.pie(data, 
             values='tip', 
             names='day',hole = 0.5)
figure.show()

figure = px.pie(data, 
             values='tip', 
             names='sex',hole = 0.5)
figure.show()

figure = px.pie(data, 
             values='tip', 
             names='smoker',hole = 0.5)
figure.show()

figure = px.pie(data, 
             values='tip', 
             names='time',hole = 0.5)
figure.show()

# Waiter Tips Prediction Model

data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
data["smoker"] = data["smoker"].map({"No": 0, "Yes": 1})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data["time"] = data["time"].map({"Lunch": 0, "Dinner": 1})
data.head()

x = np.array(data[["total_bill", "sex", "smoker", "day", 
                   "time", "size"]])
y = np.array(data["tip"])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)

# features = [[total_bill, "sex", "smoker", "day", "time", "size"]]
features = np.array([[24.50, 1, 0, 0, 1, 4]])

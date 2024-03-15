import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

st.title("Welcome to our group project for ECS 171!")

st.markdown("This is our front end which will display our models")

st.markdown("Let's say you are a student who wants to know how crowded the gym will be.")
st.markdown("Using our modeling, we will help you find out given an hour and day of the week!")

st.title("Let's begin!")
st.markdown("First let's show our data which looks something like this")
# Read in data
df = pd.read_csv('gym_data.csv')
df_head = df.head()

st.dataframe(df_head)


# Filter to most important variables based on above scores and previous EDA
# Use only timestamp since timestamp = hour
X = df[['day_of_week', 'hour', 'temperature']]
y = df['number_people']

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
X_rescaled = scaler.fit_transform(X)
X_rescaled = pd.DataFrame(data = X_rescaled, columns = X.columns)

# Split data to train and test set 80:20
X_train,X_test,y_train,y_test=train_test_split(X_rescaled,y,test_size=0.2,random_state=100)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train,y_train)

# Get predictions
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# st.markdown('Test MSE:',mean_squared_error(y_test, y_test_pred))
# st.markdown('Train MSE:',mean_squared_error(y_train, y_train_pred))

# Change negative predictions to 0, there cannot be negative people at the gym
y_test_pred[y_test_pred < 0] = 0
y_train_pred[y_train_pred < 0 ] = 0

# Plot
fig_1, ax_1 = plt.subplots()
ax_1.scatter(X_test['hour'], y_test_pred)

# Compare to actual data
fig_2, ax_2 = plt.subplots()
ax_2.scatter(X_test['hour'], y_test)

st.markdown("Here is what the our linear regression plot looks like:")
st.pyplot(fig_1)
st.markdown("Here is what the original dataset looks like:")
st.pyplot(fig_2)
st.markdown("The Test MSE of this data is 306.22")
st.markdown("& The Train MSE of this data is 304.49")


st.markdown("As you can see, our linear regression model is pretty good, but we decided to make some better models.")


st.markdown("We then made a neural network, and after a few iterations found a good learning rate as well with this code:")
code = '''
mlp = MLPRegressor(learning_rate_init = 0.03, max_iter=500)
mlp.fit(X_train, y_train)

y_test_pred_nn = mlp.predict(X_test)
y_train_pred_nn = mlp.predict(X_train)

# Change all predictions less than 0 to be 0
y_test_pred_nn[y_test_pred_nn < 0] = 0
y_train_pred_nn[y_train_pred_nn < 0 ] = 0'''

st.code(code, language='python')

# With random search tuned hyperparameters

st.markdown("This led us to this graph:")
st.image('Screenshot 2024-03-13 at 6.35.28 PM.png', caption='neural network graph')
# mlp = MLPRegressor(learning_rate_init = 0.03, max_iter=500)
# mlp.fit(X_train, y_train)

# y_test_pred_nn = mlp.predict(X_test)
# y_train_pred_nn = mlp.predict(X_train)

# # Change all predictions less than 0 to be 0
# y_test_pred_nn[y_test_pred_nn < 0] = 0
# y_train_pred_nn[y_train_pred_nn < 0 ] = 0

# fig_3, ax_3 = plt.subplots()
# ax_3.scatter(X_test['hour'], y_test_pred_nn)

# st.pyplot(fig_3)

st.markdown("This neural network had Test MSE of 232.09 & Train MSE of 228.56")
st.markdown("This is already better representation than linear regression!")

st.markdown("")
st.markdown("We then made polynomial regression graphs as well but realized that although they had good representation, they began to overfit around 6-7 degrees")
st.markdown("You can find these graphs in our Google Colab to see overfitting more clearly! The MSE also didn't get much lower than our Neural Network")
#st.markdown("Note: These would take too long to run in our frontend so we have screenshots of them instead:")

st.image('Screenshot 2024-03-03 at 3.50.24 PM.png', caption='example of degree 4')


st.markdown("Now we will make predictions using our model of choice which was degree 5 polynomial regression")

# Define the degree of the polynomial
degree = 5

poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

y_train_pred_poly = poly_model.predict(X_poly_train)
y_test_pred_poly = poly_model.predict(X_poly_test)

# Ensuring we don't have negative predictions
y_train_pred_poly[y_train_pred_poly < 0] = 0
y_test_pred_poly[y_test_pred_poly < 0] = 0

# Calculate and display MSE
test_mse = mean_squared_error(y_test, y_test_pred_poly)
train_mse = mean_squared_error(y_train, y_train_pred_poly)
st.write(f"Test MSE for degree {degree}: {test_mse:.2f}")
st.write(f"Train MSE for degree {degree}: {train_mse:.2f}")

# Plotting the results
fig_poly, ax_poly = plt.subplots()
ax_poly.scatter(X_test['hour'], y_test_pred_poly, label='Predictions')
#ax_poly.scatter(X_test['hour'], y_test, label='Actual Data', alpha=0.5)
ax_poly.legend()
ax_poly.set_title(f"Polynomial Regression (Degree {degree}) Predictions")
ax_poly.set_xlabel('Hour of the Day')
ax_poly.set_ylabel('Number of People')

st.pyplot(fig_poly)


st.title("Let's do a short demo!")
st.markdown("give an input of hour and day of the week, and we will predict how crowded the gym will be!")

hour_of_day = st.selectbox(
    'When will you go to the gym?',
    ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'))

st.write('You will go at:', hour_of_day, "o'clock")

day_of_week = st.selectbox(
    'What day of the week?',
    ('sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'))

st.write('You will go on:', day_of_week)

expected_temp = st.slider('What is the expected temperature? (F)', 60, 85, 70)
st.write("You expecting a temperature of:", expected_temp)

day_to_int = {
        'sunday': 0,
        'monday': 1,
        'tuesday': 2,
        'wednesday': 3,
        'thursday': 4, 
        'friday': 5,
        'saturday': 6
}

day_of_week_int = day_to_int[day_of_week]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)

user_input = [[day_of_week_int, int(hour_of_day), expected_temp]]
input_rescaled = scaler.transform(user_input)
input_poly = poly.transform(input_rescaled)
num_people = poly_model.predict(input_poly)

if num_people < 0:
    num_people = 0
    st.write("Based off of your input, we would expect: 0 people to be at the gym, please change the time, day, or temp")
else:
    st.write("Based off of your input, we would expect:", num_people, "People to be at the gym")





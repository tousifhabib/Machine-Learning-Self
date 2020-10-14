import pandas as pd
from sklearn import linear_model
import tkinter as tk 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

data = pd.read_csv("D:\E-By Design\Machine Learning\Real estate valuation data set.csv")
data = data.dropna()

df = pd.DataFrame(data,columns=['X1 transaction date','X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores','X5 latitude','X6 longitude','Y house price of unit area']) 

X = df[['X1 transaction date','X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores','X5 latitude','X6 longitude']].astype(float)
Y = df['Y house price of unit area'].astype(float)

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# tkinter GUI
root= tk.Tk()

canvas1 = tk.Canvas(root, width = 200, height = 200)
canvas1.pack()

# with sklearn
Intercept_result = ('Intercept: ', regr.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')
canvas1.create_window(260, 220, window=label_Intercept)

# with sklearn
Coefficients_result  = ('Coefficients: ', regr.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')
canvas1.create_window(260, 240, window=label_Coefficients)



# prediction with sklearn
New_Transaction_Date = 2013.417
New_House_Age = 20.3
New_Distance_To_The_Nearest_MRT_Station =  287.6025
New_Number_Of_Convenience_Stores = 6
New_Latitude = 24.98042
New_Longitude = 121.5423

print ('Predicted Stock House Per Unit Price: \n', regr.predict([[New_Transaction_Date ,New_House_Age, New_Distance_To_The_Nearest_MRT_Station, New_Number_Of_Convenience_Stores, New_Latitude, New_Longitude]]))
       
# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)
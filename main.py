import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import sklearn.model_selection as sk 
import sklearn.linear_model as lm

# Original Dataset : 
# Dollar : 16600 rows X 7 columns 
# VIX : 8994 rows X 7 columns 

main_df=pd.read_csv("combined.csv")

print(main_df)

x=main_df["Vix Close"]
y=main_df["Dollar Close"]

# Creating training and testing Data 
x_train, x_test, y_train, y_test = sk.train_test_split(x,y,train_size=0.8,random_state=15)

x_train=np.array(x_train).reshape(-1,1)
x_test=np.array(x_test).reshape(-1,1)

# Linear Regression 
lr = lm.LinearRegression()
lr.fit(x_train,y_train)

prediction= lr.predict(x_test)

print("Linear Regression Score "+str(lr.score(x_test,y_test)))

# Huber Regression 
hr = lm.HuberRegressor(epsilon=1.0)
hr.fit(x_train,y_train)

prediction2= hr.predict(x_test)

print(" Huber Regression Score "+str(hr.score(x_test,y_test)))


# Plotting Models
plt.plot(x_test,prediction,color="red",label="Model")
plt.scatter(x_test,y_test,label="Test Data")
plt.show()

plt.plot(x_test,prediction2,color="red")
plt.scatter(x_test,y_test)
plt.show()



# Plotting the VIX and Dollar data. X-axis : Dates, Y-axis: Close price
#main_df.plot(x="Date",y="Vix Close",title="Close VIX data")
#main_df.plot(x="Date",y="Dollar Close",title="Close Dollar data")

# Plotting the VIX and Dollar data. X-axis : Dates, Y-axis: Change price
#main_df.plot(x="Date",y="Vix Change",title="Change VIX data")
#main_df.plot(x="Date",y="Dollar Change",title="Change Dollar data")

# Plotting the frequencies of the data 
#main_df.plot(x="Date",y="Vix Close",title="Frquency VIX data",kind="hist")
#main_df.plot(x="Date",y="Dollar Close",title="Frequency Dollar data",kind="hist")

#main_df.plot(x="Date",y="Vix Change",title="Frquency VIX Change data",kind="hist")
#main_df.plot(x="Date",y="Dollar Change",title="Frequency Dollar Change data",kind="hist")

#main_df.plot(x="Vix Close",y="Dollar Close",title="Dollar vs Vix",kind="scatter")



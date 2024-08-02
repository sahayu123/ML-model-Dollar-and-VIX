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

#print(x)
#print(y)

x_train, x_test, y_train, y_test = sk.train_test_split(x,y,train_size=0.8)

#plt.scatter(x_train,y_train)
#plt.scatter(x_test,y_test)

x_train=np.array(x_train).reshape(-1,1)
x_test=np.array(x_test).reshape(-1,1)

#print(x_train)
#print(x_test)

lr = lm.LinearRegression()
lr.fit(x_train,y_train)

print(lr.intercept_)
print(lr.coef_)

prediction= lr.predict(x_test)

print(lr.score(x_test,y_test))

plt.plot(x_test,prediction,color="red",label="Model")
plt.scatter(x_test,y_test,label="Test Data")




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


plt.show()
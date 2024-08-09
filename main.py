import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import sklearn.model_selection as sk 
import sklearn.linear_model as lm
import math 

# Original Dataset : 
# Dollar : 16600 rows X 7 columns 
# VIX : 8994 rows X 7 columns 

main_df=pd.read_csv("combined.csv")

x=main_df["Vix Close"]
y=main_df["Dollar Close"]

# Creating training and testing Data 
x_train, x_test, y_train, y_test = sk.train_test_split(x,y,train_size=0.8,random_state=15)

x_train=np.array(x_train).reshape(-1,1)
x_test=np.array(x_test).reshape(-1,1)

# Regression Models 
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

# Quantile Regression 
qr = lm.QuantileRegressor(quantile=0.5)
qr.fit(x_train,y_train)
prediction3= qr.predict(x_test)
print(" Quantile Regression Score "+str(qr.score(x_test,y_test)))

# RANCAS Regression 
rr = lm.RANSACRegressor(random_state=6)
rr.fit(x_train,y_train)
prediction4= rr.predict(x_test)
print(" RANSAC Regression Score "+str(rr.score(x_test,y_test)))

# Logarithmic Regression 
logs_vix=list()
logs_dollar=list()
for index, row in main_df.iterrows():
    logs_vix.append(math.log(main_df["Vix Close"][index]))
    logs_dollar.append(math.log(main_df["Dollar Close"][index]))

log_x=pd.DataFrame({"Vix Close":logs_vix})
log_y=pd.DataFrame({"Dollar Close":logs_dollar})

log_x_train, log_x_test, log_y_train, log_y_test = sk.train_test_split(log_x,log_y,train_size=0.8,random_state=15)

log_x_train=np.array(log_x_train).reshape(-1,1)
log_x_test=np.array(log_x_test).reshape(-1,1)

logr = lm.LinearRegression()
logr.fit(log_x_train,log_y_train)
prediction5= logr.predict(log_x_test)
print("Logarithmic Regression Score "+str(logr.score(log_x_test,log_y_test)))

# Linear Regression on First 30 VIX values 
under_30_vix=list()
under_30_dollar=list()

for index, row in main_df.iterrows():
    if main_df["Vix Close"][index]<31:
        under_30_vix.append(main_df["Vix Close"][index])
        under_30_dollar.append(main_df["Dollar Close"][index])

x_30=pd.DataFrame({"Vix":under_30_vix})
y_30=pd.DataFrame({"Dollar":under_30_dollar})

x_train30, x_test30, y_train30, y_test30 = sk.train_test_split(x_30,y_30,train_size=0.8,random_state=15)

x_train30=np.array(x_train30).reshape(-1,1)
x_test30=np.array(x_test30).reshape(-1,1)

lr30 = lm.LinearRegression()
lr30.fit(x_train30,y_train30)
prediction30= lr.predict(x_test30)
print("First 30 Linear Regression Score "+str(lr.score(x_test30,y_test30)))





# Time Series Model 

# Polynomial Regression 

#Multiple x values 

# Plotting Models
plt.plot(x_test,prediction,color="red",label="Model")
plt.scatter(x_test,y_test,label="Test Data")
plt.show()

plt.plot(x_test,prediction2,color="red")
plt.scatter(x_test,y_test)
plt.show()

plt.plot(x_test,prediction3,color="red")
plt.scatter(x_test,y_test)
plt.show()

plt.plot(x_test,prediction4,color="red")
plt.scatter(x_test,y_test)
plt.show()

plt.plot(log_x_test,prediction5,color="red")
plt.scatter(log_x_test,log_y_test)
plt.show()

plt.plot(x_test30,prediction30,color="red",label="Model")
plt.scatter(x_test30,y_test30,label="Test Data")
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



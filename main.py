import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import sklearn.model_selection as sk 
import sklearn.linear_model as lm
import math 
from sklearn.metrics import r2_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


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
    if (main_df["Vix Close"][index]<31):
        under_30_vix.append(main_df["Vix Close"][index])
        under_30_dollar.append(main_df["Dollar Close"][index])

x_30=pd.DataFrame({"Vix":under_30_vix})
y_30=pd.DataFrame({"Dollar":under_30_dollar})

x_train30, x_test30, y_train30, y_test30 = sk.train_test_split(x_30,y_30,train_size=0.8,random_state=15)

x_train30=np.array(x_train30).reshape(-1,1)
x_test30=np.array(x_test30).reshape(-1,1)

lr30 = lm.LinearRegression()
lr30.fit(x_train30,y_train30)
prediction6= lr.predict(x_test30)
print("First 30 Linear Regression Score "+str(lr.score(x_test30,y_test30)))

# Multiple Variables Linear Regression 
yesterday_list=list()
day_before_list=list()
for index, row in main_df.iterrows():
        try:
            yesterday_list.append(main_df["Vix Close"][index-1])
        except: 
            yesterday_list.append(0.0)
        try: 
            day_before_list.append(main_df["Vix Close"][index-2])
        except:
            day_before_list.append(0.0)

yesterday_list=yesterday_list[2:]
day_before_list= day_before_list[2:]
mv_x=x[2:]
mv_y=y[2:]
new_x=pd.DataFrame({"Vix Close":mv_x, "Yesterday Close":yesterday_list,"Day Before Close":day_before_list})
mul_x_train, mul_x_test, mul_y_train, mul_y_test = sk.train_test_split(new_x,mv_y,train_size=0.8,random_state=15)

lrm = lm.LinearRegression()
lrm.fit(mul_x_train,mul_y_train)
prediction7= lrm.predict(mul_x_test)

print("Multiple Variables Linear Regression "+str(r2_score(mul_y_test,prediction7)))

# Decision Tree Classifier 
signs=list()
for index, row in main_df.iterrows():
    if main_df["Dollar Change"][index] >= 0:
        signs.append(1)
    else:
        signs.append(0)
dt_y=pd.DataFrame({"Dollar Sign":signs})
print(dt_y)
print("CNT : ",cnt)


dtc_x_train, dtc_x_test, dtc_y_train, dtc_y_test = sk.train_test_split(x,dt_y,train_size=0.8,random_state=15)

dtc_x_train=np.array(dtc_x_train).reshape(-1,1)
dtc_x_test=np.array(dtc_x_test).reshape(-1,1)

dtc = DecisionTreeClassifier()
dtc.fit(dtc_x_train,dtc_y_train)

dtc_y_pred=dtc.predict(dtc_x_test)

print(classification_report(dtc_y_test,dtc_y_pred))
print(confusion_matrix(dtc_y_test,dtc_y_pred))




# Plotting Models
if False :
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

    plt.plot(x_test30,prediction6,color="red",label="Model")
    plt.scatter(x_test30,y_test30,label="Test Data")
    plt.show()

    plt.scatter(mul_y_test,prediction7)
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



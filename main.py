import pandas as pd 
import matplotlib.pyplot as plt
dollar_df = pd.read_csv("dollar.csv")
vix_df = pd.read_csv("vix.csv")

# Dropping the Volume Data Column, NAN values, and duplicate values 
def clean_data(df):
    df= df.drop(columns=["Volume"])
    df= df.dropna()
    df= df.drop_duplicates()
    return df

vix_df=clean_data(vix_df)
dollar_df=clean_data(dollar_df)

#Adding the Change Column 
def add_change(df):
    change_list=list()
    for index, row in df.iterrows():
        try:
            change_list.append(df["Close"][index]-df["Close"][index-1])
        except: 
            change_list.append(0.0)
    return change_list

vix_df["Change"]=add_change(vix_df)
dollar_df["Change"]=add_change(dollar_df)

print(vix_df)
print(dollar_df)

# Plotting the VIX and Dollar data. X-axis : Dates, Y-axis: Close price
vix_df.plot(x="Date",y="Close",title="Close VIX data")
dollar_df.plot(x="Date",y="Close",title="Close Dollar data")

# Plotting the VIX and Dollar data. X-axis : Dates, Y-axis: Change price
vix_df.plot(x="Date",y="Change",title="Change VIX data")
dollar_df.plot(x="Date",y="Change",title="Change Dollar data")

plt.show()







# number of data 
# clean the NAN values
# calculate change from  yestersterday, use close values. Plot it. 
# calculate correlation. Use some sort of correlation metric. Compute correlation. Group up by month or year, and compute correlation. 
# use 50 days data. If there is daily seasonality, it would help smooth out. 
# check if there are outlier data 
# check when VIX or US dollar index are abnormally high. 
# Plot time series data. 
# filter out everything before VIX index. 
# create one dataset with everything 

'''
High priority: comoputing total data amount
(number of rows and columns in each csv) 
Computing the number of missing values and duplicate entries
Moderate priority: making a histogram of the values of usdx and vix
Low priority: plot over time of vix and usdx
outlier analysis
(when are things the highest/lowest)
Grouping by day and computing correlations
Computing the lagged differences and plotting histogram
'''
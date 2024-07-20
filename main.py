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

vix_df=vix_df.reset_index(drop=True)
dollar_df=dollar_df.reset_index(drop=True)

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

# Fitting the Dollar in the range of the vix
start=int()
end=int()
index=dollar_df.loc[dollar_df["Date"]=="1990-01-02"]
for index,row in index.iterrows():
    start=index
index=dollar_df.loc[dollar_df["Date"]=="2024-06-21"]
for index,row in index.iterrows():
    end=index

# Start : 4792
# End : 13581

dollar_df=dollar_df[start:(end+1)]
dollar_df=dollar_df.reset_index(drop=True)

print(vix_df)
print(dollar_df)


# Plotting the VIX and Dollar data. X-axis : Dates, Y-axis: Close price
vix_df.plot(x="Date",y="Close",title="Close VIX data")
dollar_df.plot(x="Date",y="Close",title="Close Dollar data")

# Plotting the VIX and Dollar data. X-axis : Dates, Y-axis: Change price
vix_df.plot(x="Date",y="Change",title="Change VIX data")
dollar_df.plot(x="Date",y="Change",title="Change Dollar data")

plt.show()







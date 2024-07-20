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
def print_start_and_end(df):
    print(df.loc[df["Date"]=="1990-01-02"])   
    print(df.loc[df["Date"]=="2024-06-21"])      
# Start : 4792
# End : 13581

dollar_df=dollar_df[4792:13582]
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







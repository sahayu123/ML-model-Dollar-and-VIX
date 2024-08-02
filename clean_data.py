import pandas as pd 
import matplotlib.pyplot as plt
import time 
start_time=time.time()
dollar_df = pd.read_csv("dollar.csv")
vix_df = pd.read_csv("vix.csv")


# Dropping the Volume Data Column, NAN values, and duplicate values 
def clean_data(df):
    df= df.drop(columns=["Volume","Adj Close"])
    df= df.dropna()
    df= df.drop_duplicates()
    return df

vix_df=clean_data(vix_df)
dollar_df=clean_data(dollar_df)

vix_df=vix_df.reset_index(drop=True)
dollar_df=dollar_df.reset_index(drop=True)

print(vix_df)
print(dollar_df)


date_list=list()

dollar_open_list=list()
dollar_high_list=list()
dollar_low_list=list()
dollar_close_list=list()



# Getting the Dollar and VIX on the same date 
for index, row in dollar_df.iterrows():
  if (vix_df["Date"]==row["Date"]).any():
    date_list.append(row["Date"])
    dollar_open_list.append(row["Open"])
    dollar_high_list.append(row["High"])
    dollar_low_list.append(row["Low"])
    dollar_close_list.append(row["Close"])
  
  else: 
    continue 

vix_open_list=list()
vix_high_list=list()
vix_low_list=list()
vix_close_list=list()


for index,row in vix_df.iterrows():
    if row["Date"] in date_list:
        vix_open_list.append(row["Open"])
        vix_high_list.append(row["High"])
        vix_low_list.append(row["Low"])
        vix_close_list.append(row["Close"])

# combining datasets 
combined_df= pd.DataFrame({"Date":date_list,"Vix Open":vix_open_list,"Vix High": vix_high_list,
"Vix Low":vix_low_list,"Vix Close":vix_close_list, "Dollar Open":dollar_open_list,
"Dolar High": dollar_high_list,"Dollar Low":dollar_low_list, "Dollar Close":dollar_close_list,
})



#Adding the Change Column 
def add_change(df,label):
    change_list=list()
    for index, row in df.iterrows():
        try:
            change_list.append(df[label][index]-df[label][index-1])
        except: 
            change_list.append(0.0)
    return change_list

combined_df["Vix Change"]=add_change(combined_df,"Vix Close")
combined_df["Dollar Change"]=add_change(combined_df, "Dollar Close")

# exporting to csv file 
combined_df.to_csv("combined.csv")

# measuring program runtime 
end_time=time.time()
print(end_time-start_time)








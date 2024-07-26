import pandas as pd 
import matplotlib.pyplot as plt

main_df=pd.read_csv("combined.csv")

print(main_df)

# Plotting the VIX and Dollar data. X-axis : Dates, Y-axis: Close price
main_df.plot(x="Date",y="Vix Close",title="Close VIX data")
main_df.plot(x="Date",y="Dollar Close",title="Close Dollar data")

# Plotting the VIX and Dollar data. X-axis : Dates, Y-axis: Change price
main_df.plot(x="Date",y="Vix Change",title="Change VIX data")
main_df.plot(x="Date",y="Dollar Change",title="Change Dollar data")

# Plotting the frequencies of the data 
main_df.plot(x="Date",y="Vix Close",title="Frquency VIX data",kind="hist")
main_df.plot(x="Date",y="Dollar Close",title="Frequency Dollar data",kind="hist")



plt.show()
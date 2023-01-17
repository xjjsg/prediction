import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

df = pd.read_csv("data.csv")
df.drop(columns=["date"], inplace=True)
df.insert(df.shape[1], "index", 1)
colum = df.columns.delete(49)
for i in range(len(colum)):
    vif_list = [vif(df[colum].values, index) for index in range(len(colum))]
    maxvif = max(vif_list)
    print("Max VIF value is ", maxvif)
    drop_index = vif_list.index(maxvif)
    print("For Independent variable", colum[drop_index])
    if maxvif > 10:
        print("Deleting", colum[drop_index])
        colum = colum.delete(drop_index)
        print("Final Independent_variables ", colum)
    else:
        break

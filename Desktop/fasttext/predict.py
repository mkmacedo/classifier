import pandas as pd
import fasttext as ft

# here you load the csv into pandas dataframe
df = pd.read_csv('smallnormalizar.csv', sep=',')

# here you load your fasttext module
model = ft.load_model("produtos.bin")

# line by line, you make the predictions and store them in a list
predictions = []
for line in df['produto']:
    pred_label = model.predict(line, k=5)
    predictions.append(pred_label)

# you add the list to the dataframe, then save the datframe to new csv

df['prediction'] = predictions
print(df.to_string())


df.to_csv('produtods_predict.csv', sep=',', index=False)

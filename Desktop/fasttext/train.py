## ConnectCom Text Classifier SiteMercado
## Developed by Fabio Covolo Mazzo
import pandas as pd
import fasttext

# Press the green button in the gutter to run the script.
print('ConnectCom Classifier Version 4.0  - Site Mercado')
print("Transform categories and subcategories in labels")

csvFileTrain = pd.read_csv("produtos_normalizados.csv.train", low_memory=False)
#columnsToKeep = ['categoria', 'catsub', 'departamento', 'produto']

columnsToKeep = ['categoria', 'produto']

newFileTrain = csvFileTrain[columnsToKeep]
newFileTrain.to_csv("produtos.train", index=False)

csvFileTrainValid = pd.read_csv("produtos_normalizados.csv.valid", low_memory=False)
newFileValid = csvFileTrainValid[columnsToKeep]
newFileValid.to_csv("produtos.valid", index=False)

model = fasttext.train_supervised(input="produtos.train")
model.test("produtos.valid")

model.save_model("produtos.bin")

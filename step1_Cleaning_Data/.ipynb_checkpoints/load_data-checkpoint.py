import pandas as pd

url = "https://raw.githubusercontent.com/patilpranjal/Auto-Grader-for-short-answer-Question-using-Siamese-Neural-Network/master/Dataset.csv"
df = pd.read_csv(
    url,
    index_col= 0
)

lst = list(df.referenceAnswer.value_counts().index)

temp = pd.DataFrame(lst, columns = ["answer"])

temp['group'] = temp.index

df = pd.merge(
    left = df,
    left_on = "referenceAnswer",
    right = temp,
    right_on = 'answer'
)

df = df.drop('answer', axis = 1)

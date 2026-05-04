import pandas as pd
import numpy as np

df = pd.read_csv("Results\\eval_recommendations_with_relevance.csv")

queries_and_seeds = {}
for input_type, input_text in zip(df["input_type"], df["input_text"]):
    if input_text not in queries_and_seeds:
        queries_and_seeds[input_text] = input_type

relevance = np.reshape(df["relevant"], (len(queries_and_seeds), 5))

precision = [float(sum(row)/5) for row in relevance]

MRR = 0
for row in relevance:
    if (sum(row) == 0):
        MRR += 0
        continue
    for i in range(5):
        if row[i] == 1:
            MRR += float(1 / (i + 1))
            break
MRR /= len(relevance)

nDCG = []
for row in relevance:
    DCG = 0
    for i in range(5):
        DCG += row[i] / np.log2(i + 2)
    
    ideal = 0
    for i in range(sum(row)):
        ideal += 1 / np.log2(i + 2)
    
    nDCG.append(float(DCG/ideal)) if ideal != 0 else nDCG.append(0)

print("Precision: ", precision)
print("Mean Precision: ", sum(precision)/len(precision))
print("MRR: ", MRR)
print("nDCG: ", np.round(nDCG, 3).tolist())
print("Mean nDCG: ", sum(nDCG)/len(nDCG))
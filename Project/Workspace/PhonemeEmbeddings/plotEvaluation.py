import numpy as np 
import pickle  
import matplotlib.pyplot as plt
import seaborn
from pandas import DataFrame
performances = pickle.load(open("performances.pkl","rb"))

dim_embeddingInterval = [2,5,
                         10,20,50,100,150,200,400
                         ]
contextWindowInterval = [1,2,
                         3,4,5,8
                         ]

negativeSamplingInterval = list(range(1,21))

#get max
max = 0
for sg in performances:
    for dim_embedding in performances[sg]:
        for contextWindow in performances[sg][dim_embedding]:
            for hs in performances[sg][dim_embedding][contextWindow]:
                if hs == 0:
                    for k_negative in performances[sg][dim_embedding][contextWindow][hs]:
                        acc = np.mean(performances[sg][dim_embedding][contextWindow][hs][k_negative]["ACCURACY"])
                        if acc > max :
                            print(sg,dim_embedding,contextWindow,hs,k_negative)
                            print(acc)  
                            max = acc               
                else:
                    acc = np.mean(performances[sg][dim_embedding][contextWindow][hs]["ACCURACY"])
                    if acc > max :
                        print(sg,dim_embedding,contextWindow,hs)
                        print(acc)
                        max = acc

                        

# sg = 0
# hs = 0
# max = 0
# dim_embedding = 400
# # context window vs  k_negative
# df = DataFrame(np.zeros((len(contextWindowInterval),len(negativeSamplingInterval))),
#                columns = negativeSamplingInterval,
#                index = contextWindowInterval)
# for contextWindow in performances[sg][dim_embedding]:
#     for k_negativeSampling in performances[sg][dim_embedding][contextWindow][hs]:
#         acc = np.mean(performances[sg][dim_embedding][contextWindow][hs][k_negativeSampling]["ACCURACY"])
#         print(contextWindow,k_negativeSampling)
#         print("ACC",acc)
#         df[k_negativeSampling][contextWindow] = acc
# 
# seaborn.heatmap(df, annot=True)
# plt.show()
#     

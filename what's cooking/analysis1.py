import json
import pandas as pd
import numpy as np

with open('train.json') as data_file:    
    data = json.load(data_file)

print data[0]
print 'Length of dataset:', len(data)
ingred_list = set()
cuisine_list = set()

for item in data:
    ingred_list |= set(item['ingredients'])
    cuisine_list.add(item['cuisine'])

ingred_list = list(ingred_list)
cuisine_list = list(cuisine_list)

print 'List of cuisines:', len(cuisine_list)
print cuisine_list
print '\nList of ingredients', len(ingred_list)

ingred_dict = dict()
for i, ingredient in enumerate(ingred_list):
    ingred_dict[ingredient] = i

feature_df_np = np.zeros((len(data), len(ingred_list)))
label_df_np = []
for i, item in enumerate(data):
    label_df_np.append(item['cuisine'])
    for ingredient in item['ingredients']:
        feature_df_np[i][ingred_dict[ingredient]] = 1

df_pd = pd.DataFrame(feature_df_np, columns=ingred_list)
df_pd['cuisine'] = label_df_np
# print df_pd.head()


# Salt occurs the highest number of times:18048
'''Top 21:
0 salt 18048
1 onions 7972
2 olive oil 7971
3 water 7457
4 garlic 7380
5 sugar 6434
6 garlic cloves 6236
7 butter 4847
8 ground black pepper 4784
9 all-purpose flour 4632
10 pepper 4438
11 vegetable oil 4385
12 eggs 3388
13 soy sauce 3296
14 kosher salt 3113
15 green onions 3078
16 tomatoes 3058
17 large eggs 2948
18 carrots 2814
19 unsalted butter 2782
20 extra-virgin olive oil 2747
'''
sum_list = list(feature_df_np.sum(axis=0))
freq_list = sorted(sum_list, reverse=True)
names = list(df_pd)
freq_dict = dict()

for i, count in enumerate(freq_list):
    freq_dict[names[sum_list.index(count)]] = int(sum_list[sum_list.index(count)])
    print i, names[sum_list.index(count)], int(sum_list[sum_list.index(count)])
    sum_list[sum_list.index(count)] = -1

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en import English

# %%
df = pd.read_json(r'C:\Users\Simon\CSCI370\items.json')      #Importing .json file from twitter scraper to pandas dataframe

# %%
for i in range(2):
    for n in df.name:
        new = n.replace('\n','')
        person = np.array([token for token in nlp(new) if token.ent_type_ == 'PERSON'])
        for token in person:
            strtoken = str(token)
            if not strtoken.strip():
                person = np.delete(person,np.argwhere(person==token))
        if person.size == 0:
            df = df.drop(df[df.name == n].index.values.astype(int)[0])
        else:
            person = [token.orth_ for token in person]
            df = df.replace({'name' : n}, ' '.join(person))
df.reset_index(drop=True,inplace=True)

# %%
G = nx.Graph()
G.add_nodes_from(df.name.tolist())

for index,i in enumerate(G.nodes()):
    temp = list(G.nodes)
    temp.pop(index)
    for j in temp:
        G.add_edge(i,j)
        
nx.draw(G,with_labels = True)
plt.show()

# %%


# %%

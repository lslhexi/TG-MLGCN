import numpy as np
import torch
from torchtext.vocab import GloVe

glove_path = r'D:\yolo\ML-GCN-master\glove.6B\resultFile'

glove_name = '6B'
vector_dim = '300'
glove_vectors = GloVe(name=glove_name, dim=vector_dim, cache=glove_path)
num_classes=20
word_embding=np.zeros((num_classes,int(vector_dim)))
category=0
phrase = [["breeding","pond"],["round",'farmland'],['square','farmland'],['woodland'],['hills'],['ridge'],['snow','fort'],["beach"],["sea"],["lake"],["river"],['mountain','settlements'],['suburban','settlements'],['urban','settlements'],['idle','land'],["road"],["desert"],['uncultivated','land'],["cloud"],["shadow"]]
#phrase=[["round",'farmland']]
for words in phrase:
    word_sum = torch.zeros(300, )
    #print(word_sum)
    for word in words:
        if word in glove_vectors.stoi:
            word_vector = glove_vectors.vectors[glove_vectors.stoi[word]]
            print(f"Word: {word}")
            word_sum=torch.add(word_sum,word_vector)
        else:
            print(f"The word '{word}' is not in the vocabulary.")
    word_embding[category]=word_sum
    category+=1
np.save(r'D:\yolo\ML-GCN-master\word_embding'+"/word_embding.npy",word_embding)

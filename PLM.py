

import torch
import numpy as np
import transformers
from transformers import BertModel,RobertaModel,GPT2Model,AlbertModel
from scipy.spatial.distance import cosine

address = ''
address1 =address +'Robert.txt'
file1 = open(address1,'w')

def cos_similarity_matrix(matrix):
    num_rows = matrix.shape[0]
    similarity_matrix = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(i, num_rows):
            similarity_matrix[i, j] = 1 - cosine(matrix[i], matrix[j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return abs(similarity_matrix)
def cos_similarity_matrix_row(matrix):
    num_rows = matrix.shape[1]
    similarity_matrix = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(i, num_rows):
            similarity_matrix[i, j] = 1 - cosine(matrix[:,i], matrix[:,j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return abs(similarity_matrix)
def Mean(matrix):
    number = matrix.shape[0]
    matrix_mean = matrix.mean()
    mean_out = (matrix_mean - (1/number))*(number/(number-1))
    return mean_out
def Gram_matrix(matrix):
    matrix_transpose = np.transpose(matrix)
    Gram_matrix = np.dot(matrix,matrix_transpose)
    return Gram_matrix



#bert_model = BertModel.from_pretrained('bert-base-uncased')
Robert_model = RobertaModel.from_pretrained('roberta-base')
#GPT2_model = GPT2Model.from_pretrained('gpt2')
#Albert_model = AlbertModel.from_pretrained('albert-base-v2')

for name, param in Robert_model.named_parameters():
    if name == 'encoder.layer.0.attention.self.query.weight':
        print(f"Parameter name: {name}")
        file1.writelines(f"Parameter name: {name}"+'\n')  
        print(f"Parameter value: {param.data.size()}")
        attention_query_cos = cos_similarity_matrix_row(param.data)
        Gram = Gram_matrix(param.data)
        np.savetxt(address+'Gram_trained.txt', Gram, fmt='%.2f')
        np.savetxt(address+'cos_matrix_Robert_Q.txt', attention_query_cos, fmt='%.2f')   
        #print(attention_query_cos.mean())
        print(Mean(attention_query_cos))
        file1.writelines(str(Mean(attention_query_cos))+'\n') 
        print(attention_query_cos.shape)
        print('='*50)
        
    if name == 'encoder.layer.0.attention.self.key.weight':
        print(f"Parameter name: {name}")
        print(f"Parameter value: {param.data.size()}")
        file1.writelines(f"Parameter name: {name}"+'\n')  
        attention_key_cos = cos_similarity_matrix_row(param.data)
        np.savetxt(address+'cos_matrix_Robert_K.txt', attention_query_cos, fmt='%.2f')   
        #print(attention_key_cos.mean())
        print(Mean(attention_key_cos))
        file1.writelines(str(Mean(attention_key_cos))+'\n') 
        print('='*50)        
    if name == 'encoder.layer.0.attention.self.value.weight':
        print(f"Parameter name: {name}")
        print(f"Parameter value: {param.data.size()}")
        file1.writelines(f"Parameter name: {name}"+'\n')  
        attention_value_cos = cos_similarity_matrix_row(param.data)
        np.savetxt(address+'cos_matrix_Robert_V.txt', attention_query_cos, fmt='%.2f')   
        #print(attention_value_cos.mean())
        print(Mean(attention_value_cos))
        file1.writelines(str(Mean(attention_value_cos))+'\n') 
        print('='*50)
    if name == 'encoder.layer.0.attention.output.dense.weight':
        print(f"Parameter name: {name}")
        print(f"Parameter value: {param.data.size()}")
        file1.writelines(f"Parameter name: {name}"+'\n')  
        attention_output_dense_cos = cos_similarity_matrix(param.data)
        #print(attention_output_dense_cos.mean())
        print(Mean(attention_output_dense_cos))
        file1.writelines(str(Mean(attention_output_dense_cos))+'\n') 
        
        print('='*50)
    if name == 'encoder.layer.0.intermediate.dense.weight':
        print(f"Parameter name: {name}")
        print(f"Parameter value: {param.data.size()}")
        file1.writelines(f"Parameter name: {name}"+'\n')  
        attention_intermediate_dense_cos = cos_similarity_matrix(param.data)
        #print(attention_intermediate_dense_cos)
        print(attention_intermediate_dense_cos.shape)
        print(Mean(attention_intermediate_dense_cos))
        file1.writelines(str(Mean(attention_intermediate_dense_cos))+'\n') 
        print('='*50)
    if name == 'encoder.layer.0.output.dense.weight':
        print(f"Parameter name: {name}")
        print(f"Parameter value: {param.data.size()}")
        file1.writelines(f"Parameter name: {name}"+'\n')  
        attention_output_dense_cos = cos_similarity_matrix(param.data)
        #print(attention_output_dense_cos)
        print(attention_output_dense_cos.shape)
        print(Mean(attention_output_dense_cos))
        file1.writelines(str(Mean(attention_output_dense_cos))+'\n') 
        print('='*50)

file1.close()





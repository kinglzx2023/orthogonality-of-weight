

from peft import PeftModel
from modelscope import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import numpy as np
from scipy.spatial.distance import cosine
import torch.nn.functional as F




address =''
address_1 = address+'cos_sim_lora.txt'



def cos_similarity_matrix_row(matrix):
    num_rows = matrix.shape[0]
    similarity_matrix = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(i, num_rows):
            similarity_matrix[i, j] =  torch.abs(F.cosine_similarity(matrix[i], matrix[j], dim=0))
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return similarity_matrix
def cos_similarity_matrix_column(matrix):
    num_column = matrix.shape[1]
    similarity_matrix = np.zeros((num_column, num_column))
    for i in range(num_column):
        for j in range(i, num_column):
            similarity_matrix[i, j] = 1 - cosine(matrix[:,i], matrix[:,j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return abs(similarity_matrix)
def Mean(matrix):
    number = matrix.shape[0]
    matrix_mean = matrix.mean()
    mean_out = abs((matrix_mean - (1/number))*(number/(number-1)))
    return mean_out
def Gram_matrix_row(matrix):
    matrix_transpose = np.transpose(matrix)
    Gram_matrix = np.dot(matrix,matrix_transpose)
    return Gram_matrix
def Gram_matrix_column(matrix):
    matrix_transpose = np.transpose(matrix)
    Gram_matrix = np.dot(matrix_transpose,matrix)
    return Gram_matrix
def column_block_matrix(matrix, block_size):
    n, m = matrix.shape
    num_blocks = n // block_size
    blocks = []
    for i in range(num_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        block = matrix[start:end,:]
        blocks.append(block)
    
    return blocks
Matrix = ['Q','K','V']

file1 = open(address_1,'w')

def merge_lora_to_base_model():
    model_name_or_path = ''
    adapter_name_or_path = ''
    save_path = ''

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False if config.model_type == 'llama' else True
    )
    model1 = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto',
        # device_map='cuda'
    )

    weight_params = [name for name, param in model1.named_parameters() if "weight" in name]
    for name, param in model1.named_parameters():
        if name == 'transformer.h.8.attn.c_attn.weight_111':
            print(f"Parameter name: {name}")
            #file1.writelines(f"Parameter name: {name}"+'\n')  
            print(f"Parameter value: {param.data.size()}")
            atten_blocks = column_block_matrix(param.cpu().data, 4096)
            for index,block in enumerate(atten_blocks):
                print(f"Parameter value: {block.size()}")
                cos_sim_column_end_1st=cos_similarity_matrix_row(block)
                cos_sim_row_end_1st = cos_similarity_matrix_row(block.t())
                mean_cos_sim_row_end_1st = Mean(cos_sim_row_end_1st)
                mean_cos_sim_column_end_1st = Mean(cos_sim_column_end_1st)
                print(mean_cos_sim_row_end_1st, mean_cos_sim_column_end_1st)
                file1.writelines(str(Matrix[index])+':'+ str(mean_cos_sim_row_end_1st)+','+ str(mean_cos_sim_column_end_1st)+'\n')
                print('='*50)
        if name == 'transformer.h.8.attn.c_proj.weight_11':
            print(f"Parameter name: {name}")
            block = param.cpu().data
            print(f"Parameter value: {block.size()}")
            cos_sim_column_end_1st =cos_similarity_matrix_row(block) 
            cos_sim_row_end_1st =cos_similarity_matrix_row(block.t())
            mean_cos_sim_row_end_1st = Mean(cos_sim_row_end_1st)
            mean_cos_sim_column_end_1st = Mean(cos_sim_column_end_1st)
            print(mean_cos_sim_row_end_1st, mean_cos_sim_column_end_1st)
            file1.writelines('transformer.h.8.attn.c_proj.weight'+':'+ str(mean_cos_sim_row_end_1st)+','+ str(mean_cos_sim_column_end_1st)+'\n')
            print('='*50)       
        if name == 'transformer.h.8.mlp.w1.weight_11':
            print(f"Parameter name: {name}")
            block = param.data
            print(f"Parameter value: {block.size()}")
            cos_sim_row_end_1st =cos_similarity_matrix_row(block.t())
            mean_cos_sim_row_end_1st = Mean(cos_sim_row_end_1st)
            print(mean_cos_sim_row_end_1st) 
            file1.writelines('transformer.h.8.mlp.w1.weight'+':'+ str(mean_cos_sim_row_end_1st)+'\n')
            print('='*50)     
        if name == 'transformer.h.8.mlp.w2.weight_11':
            print(f"Parameter name: {name}")
            block = param.data
            print(f"Parameter value: {block.size()}")
            cos_sim_row_end_1st =cos_similarity_matrix_row(block.t())
            mean_cos_sim_row_end_1st = Mean(cos_sim_row_end_1st)
            print(mean_cos_sim_row_end_1st)
            file1.writelines('transformer.h.8.mlp.w2.weight'+':'+ str(mean_cos_sim_row_end_1st)+'\n')
            print('='*50)     
        if name == 'transformer.h.8.mlp.c_proj.weight_11':
            print(f"Parameter name: {name}")
            block = param.data
            print(f"Parameter value: {block.size()}")
            cos_sim_column_end_1st =cos_similarity_matrix_row(block) 
            cos_sim_row_end_1st =cos_similarity_matrix_row(block.t())
            mean_cos_sim_row_end_1st = Mean(cos_sim_row_end_1st)
            mean_cos_sim_column_end_1st = Mean(cos_sim_column_end_1st)
            print(mean_cos_sim_row_end_1st, mean_cos_sim_column_end_1st)
            file1.writelines('transformer.h.8.mlp.c_proj.weight'+':'+ str(mean_cos_sim_row_end_1st)+','+ str(mean_cos_sim_column_end_1st)+'\n')
            print('='*50)     
        
        
    model = PeftModel.from_pretrained(model1, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()


    weight_params = [name for name, param in model.named_parameters() if "weight" in name]
    print('------------------------')
    print(weight_params)
    for name, param in model.named_parameters():
        if name == 'transformer.h.8.attn.c_attn.weight':
            print(f"Parameter name: {name}")
            print(f"Parameter value: {param.data.size()}")
            atten_blocks = column_block_matrix(param.cpu().data, 4096)
            for index,block in enumerate(atten_blocks):
                print(f"Parameter value: {block.size()}")
                cos_sim_column_end_1st=cos_similarity_matrix_row(block)
                cos_sim_row_end_1st = cos_similarity_matrix_row(block.t())
                mean_cos_sim_row_end_1st = Mean(cos_sim_row_end_1st)
                mean_cos_sim_column_end_1st = Mean(cos_sim_column_end_1st)
                print(mean_cos_sim_row_end_1st, mean_cos_sim_column_end_1st)
                file1.writelines(str(Matrix[index])+':'+ str(mean_cos_sim_row_end_1st)+','+ str(mean_cos_sim_column_end_1st)+'\n')
                print('='*50)
        if name == 'transformer.h.8.attn.c_proj.weight':
            print(f"Parameter name: {name}")
            block = param.data
            print(f"Parameter value: {block.size()}")
            cos_sim_row_end_1st =cos_similarity_matrix_row(block.t())
            mean_cos_sim_row_end_1st = Mean(cos_sim_row_end_1st)
            print(mean_cos_sim_row_end_1st)  
            file1.writelines('transformer.h.8.attn.c_proj.weight'+':'+ str(mean_cos_sim_row_end_1st)+'\n')
            print('='*50)       
        if name == 'transformer.h.8.mlp.w1.weight':
            print(f"Parameter name: {name}")
            block = param.data
            print(f"Parameter value: {block.size()}")
            cos_sim_row_end_1st =cos_similarity_matrix_row(block.t())
            mean_cos_sim_row_end_1st = Mean(cos_sim_row_end_1st)
            print(mean_cos_sim_row_end_1st) 
            file1.writelines('transformer.h.8.mlp.w1.weight'+':'+ str(mean_cos_sim_row_end_1st)+'\n')
            print('='*50)     
        if name == 'transformer.h.8.mlp.w2.weight':
            print(f"Parameter name: {name}")
            block = param.data
            print(f"Parameter value: {block.size()}")
            cos_sim_row_end_1st =cos_similarity_matrix_row(block.t())
            mean_cos_sim_row_end_1st = Mean(cos_sim_row_end_1st)
            print(mean_cos_sim_row_end_1st)
            file1.writelines('transformer.h.8.mlp.w2.weight'+':'+ str(mean_cos_sim_row_end_1st)+'\n')
            print('='*50)     
       
if __name__ == '__main__':
    merge_lora_to_base_model()

file1.close() 




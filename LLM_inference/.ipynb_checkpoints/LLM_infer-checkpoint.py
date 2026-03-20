import os
import yaml
from huggingface_hub import login
import pandas as pd
from tqdm import tqdm
from transformers import BitsAndBytesConfig, pipeline
import torch


def load_config(p):
    with open('config_instruction.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['parameters'][p]

os.environ["HUGGINGFACE_TOKEN"] = load_config("token")
login(token=os.environ["HUGGINGFACE_TOKEN"])

def load_model():
    model_variant = "27b-text-it"  # @param ["4b-it", "27b-it", "27b-text-it"]
    model_id = f"google/medgemma-{model_variant}"
    
    use_quantization = True  # @param {type: "boolean"}
    
    # @markdown Set `is_thinking` to `True` to turn on thinking mode. **Note:** Thinking is supported for the 27B variants only.
    is_thinking = False  # @param {type: "boolean"}
    
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if use_quantization:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    
    if "text" in model_variant:
        pipe = pipeline("text-generation", model=model_id, model_kwargs=model_kwargs)
    else:
        pipe = pipeline("image-text-to-text", model=model_id, model_kwargs=model_kwargs)

    return pipe

def prompt_generation(row, p):
    if p in ['p1', 'p5']:
        prompt = f'''Clinical Note:{row['drug_adjacent']},
    
                Input Drug: {row['standard_drug']}

                Please follow the steps.
                ''' 
    elif p in ['p2', 'p4', 'p6']:
        prompt = f''' Patient Note:{row['symptom_adjacent']}
                            
                 Please follow the steps.
                '''
    elif p == 'p3':
        prompt = f'''Patient Note:{row['symptom_adjacent']},

                Input Drug:{row['standard_drug']},
                
                Please follow the steps.
                '''
    else:
        return 'prompt_id_not_recognized'
        

    role_instruction = "You are a helpful medical assistant."
    system_instruction = f"SYSTEM INSTRUCTION: think silently if needed. {load_config(p)}"
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_instruction}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    return messages


def calling_llm(messages, pipe):
    output = pipe(messages, max_new_tokens=5500)
    response = output[0]["generated_text"][-1]["content"]
    return response

def main():
    pipe = load_model()
    for index in ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']:
        
        input_path = f'../irAE_preprocessing/data/dfs_llm/df{index[-1]}.csv'
        output_path = f'./llm_results/llm_df{index[-1]}.csv'
        
        print(f'{index}_inferring')
    
        df = pd.read_csv(input_path)
    
        if os.path.exists(output_path):
            df_out = pd.read_csv(output_path)
            completed = set(df_out.index)
            print(f"Resuming... {len(completed)} rows already done")
        else:
            df_out = df.copy()
            df_out['output'] = None
            completed = set()
    
        for i in tqdm(range(len(df))):
            if i in completed and pd.notna(df_out.loc[i, 'output']):
                continue 
            
            row = df.loc[i]
            try:
                result = calling_llm(prompt_generation(row, index), pipe)
                df_out.loc[i, 'output'] = result
                df_out.to_csv(output_path, index=False)
    
            except Exception as e:
                print(f"Error at row {i}: {e}")
                continue


if __name__ == "__main__":
    main()

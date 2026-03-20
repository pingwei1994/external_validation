import sys
sys.path.append('../src')

import pandas as pd
import mention_extraction.extract_mentions as eim
import mention_extraction.drug_ici_config
import json
import ast
import os





def preprocessing():
    config_file_json = "drug_symptom_dicts.json"
    def load_config(config_file_json):
        with open(config_file_json, 'r') as f:
            config = json.load(f)
        return config
    config = load_config(config_file_json)
    notes = pd.read_csv(f"../data/{config['ehr_file_name']}")
    drug_mentions = eim.extract_ici_mentions(notes, config['drug_dict'])
    symptom_mentions = eim.extract_symptom_mentions(notes, config['symptom_dict'])
    symptom_mentions[['report_id', 'standard_symptom', 'matched_variant',
       'start_index', 'variant_length', 'theword',
       'symptom_adjacent']].to_csv('../data/symptom_mentions.csv', index=False)
    drug_mentions[['report_id', 'standard_drug', 'matched_variant', 'start_index',
       'variant_length', 'theword', 'drug_adjacent']].to_csv('../data/drug_mentions.csv', index=False)
    drug_part = drug_mentions.groupby(['report_id', 'notes', 'standard_drug'], as_index=False).agg({'drug_adjacent':list})
    symptom_part = symptom_mentions.groupby(['report_id', 'notes', 'standard_symptom'], as_index=False).agg({'symptom_adjacent':list})
    final_data = drug_part.merge(symptom_part, on=['report_id', 'notes'], how='inner')
    final_data = final_data.merge(notes[['report_id','patient_index']], on = 'report_id', how='inner')
    final_data.to_csv('../data/final_data.csv', index=False)

    os.makedirs("../data/dfs_llm", exist_ok=True)
    data_split6(final_data)

def safe_literal_eval(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x


def data_split6(df):
    df = df.copy()  # 防止修改原始数据
    
    df['drug_adjacent'] = df['drug_adjacent'].apply(safe_literal_eval)
    df['symptom_adjacent'] = df['symptom_adjacent'].apply(safe_literal_eval)

    df1 = df[['report_id', 'standard_drug', 'drug_adjacent']] \
        .drop_duplicates(subset=['report_id', 'standard_drug']) \
        .explode('drug_adjacent')

    df2 = df[['report_id', 'standard_symptom', 'symptom_adjacent']] \
        .drop_duplicates(subset=['report_id', 'standard_symptom']) \
        .explode('symptom_adjacent')

    df3 = df[['report_id', 'standard_drug', 'standard_symptom', 'symptom_adjacent']] \
        .drop_duplicates(subset=['report_id', 'standard_drug', 'standard_symptom']) \
        .explode('symptom_adjacent')

    df4 = df2.copy()
    df5 = df[['report_id', 'standard_drug', 'drug_adjacent']] \
        .drop_duplicates(subset=['report_id', 'standard_drug'])

    df6 = df2.copy()

    df1.to_csv('../data/dfs_llm/df1.csv', index=False)
    df2.to_csv('../data/dfs_llm/df2.csv', index=False)
    df3.to_csv('../data/dfs_llm/df3.csv', index=False)
    df4.to_csv('../data/dfs_llm/df4.csv', index=False)
    df5.to_csv('../data/dfs_llm/df5.csv', index=False)
    df6.to_csv('../data/dfs_llm/df6.csv', index=False)
    
        


if __name__ == "__main__":
    preprocessing()







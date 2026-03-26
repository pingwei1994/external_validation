import os
import pandas as pd
import ast
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import re
import json

def parse_response(response_string):
    pattern = r'```json\s*(.*?)\s*```'
    match = re.search(pattern, response_string, re.DOTALL)

    if match:
        content = match.group(1).strip()
    else:
        content = response_string.strip()

    # 1️⃣ 先尝试正常 json
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2️⃣ 修正裸 None → "None"
    try:
        cleaned = content

        # 只替换 value 是 None 的情况
        # 匹配 : None  或  :None
        cleaned = re.sub(r':\s*None\b', r': "None"', cleaned)

        # 删除可能的 trailing comma
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)

        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3️⃣ 最后 fallback
    try:
        parsed = ast.literal_eval(content)

        # 如果 literal_eval 成功，也确保 None 被转成字符串
        if isinstance(parsed, dict):
            parsed = {
                k: ("None" if v is None else v)
                for k, v in parsed.items()
            }
        return parsed
    except:
        return 'parse_failed'

def dedup_parse(dfi):
    dfi = dfi.drop_duplicates().copy()
    dfi['llm_output'] = dfi['output'].apply(lambda x: parse_response(x))
    return dfi


info_order = [['Y', 'Y', 'Y', 'Y'], ['Y', 'Y', 'Y', 'N'], ['Y', 'Y', 'O', 'Y'], ['Y', 'Y', 'O', 'N'], ['Y', 'Y', 'N', 'Y'], ['Y', 'Y', 'N', 'N'],
              ['Y', 'N', 'Y', 'Y'], ['Y', 'N', 'Y', 'N'], ['Y', 'N', 'O', 'Y'], ['Y', 'N', 'O', 'N'], ['Y', 'N', 'N', 'Y'], ['Y', 'N', 'N', 'N'],
              ['N', 'Y', 'Y', 'Y'], ['N', 'Y', 'Y', 'N'], ['N', 'Y', 'O', 'Y'], ['N', 'Y', 'O', 'N'], ['N', 'Y', 'N', 'Y'], ['N', 'Y', 'N', 'N'],
              ['N', 'N', 'Y', 'Y'], ['N', 'N', 'Y', 'N'], ['N', 'N', 'O', 'Y'], ['N', 'N', 'O', 'N'], ['N', 'N', 'N', 'Y'], ['N', 'N', 'N', 'N']]


def select_priority_combination(
    df: pd.DataFrame,
    group_cols,
    combo_cols,
    priority_order,
    drop_temp_cols=True,
    unknown_strategy="error"  # "error" | "last"
):


    df = df.copy()
    priority_map = {tuple(v): i for i, v in enumerate(priority_order)}
    df["_combo_tuple"] = list(zip(*(df[col] for col in combo_cols)))
    df["_priority"] = df["_combo_tuple"].map(priority_map)
    if df["_priority"].isna().any():
        if unknown_strategy == "error":
            unknown = df.loc[df["_priority"].isna(), "_combo_tuple"].unique()
            raise ValueError(f"Found unknown combinations: {unknown}")
        elif unknown_strategy == "last":
            df["_priority"] = df["_priority"].fillna(len(priority_order))
        else:
            raise ValueError("unknown_strategy must be 'error' or 'last'")

    result = df.loc[
        df.groupby(group_cols)["_priority"].idxmin()
    ]

    if drop_temp_cols:
        result = result.drop(columns=["_combo_tuple", "_priority"])

    return result.reset_index(drop=True)


def symptom_earlier_than_drug(row):
    
    def parse_date(date_str):
        if pd.isna(date_str):
            return None
        
        if date_str in ['None', 'parse_failed', 'YYYY-MM-DD | None']:
            return None
        
        parts = str(date_str).split('-')
        if len(parts) != 3:
            return None
        
        year, month, day = parts
        
        def clean(x):
            return None if x in ['UNKNOWN', 'None'] else int(x)
        
        return (
            clean(year),
            clean(month),
            clean(day)
        )
    
    s = parse_date(row['normalized_start_date_symptom'])
    d = parse_date(row['normalized_start_date_drug'])
    
    # 如果任一无法解析
    if s is None or d is None:
        return 'N'
    
    sy, sm, sd = s
    dy, dm, dd = d
    
    # 年必须存在
    if sy is None or dy is None:
        return 'N'
    
    # 比较年
    if sy < dy:
        return 'Y'
    if sy > dy:
        return 'N'
    
    # 年相同 → 比较月
    if sm is None or dm is None:
        return 'N'
    
    if sm < dm:
        return 'Y'
    if sm > dm:
        return 'N'
    
    # 月相同 → 比较日
    if sd is None or dd is None:
        return 'N'
    
    if sd < dd:
        return 'Y'
    
    return 'N'

def ec(x):
    if x['symptom_start_date_earlier_than_drug'] == 'Y':
        return 0
    else: 
        if x['drug_negation']=='N':
            return 0
        else:
            if x['symptom_negation']=='N':
                return 0
            else:
                if x['cause_matches_input_drug']=='Y':
                    return 1
                elif x['cause_matches_input_drug']=='N':
                    return 0
                else:
                    if x['intervene_target_symtom']=='Y':
                        return 1
                    else:
                        return 2




def main():
    p1 = pd.read_csv('../LLM_inference/llm_results/llm_df1.csv')
    p2 = pd.read_csv('../LLM_inference/llm_results/llm_df2.csv')
    p3 = pd.read_csv('../LLM_inference/llm_results/llm_df3.csv')
    p4 = pd.read_csv('../LLM_inference/llm_results/llm_df4.csv')
    p5 = pd.read_csv('../LLM_inference/llm_results/llm_df5.csv')
    p6 = pd.read_csv('../LLM_inference/llm_results/llm_df6.csv')
    df1 = dedup_parse(p1)
    df1 = df1[df1['report_id']!='report_id'].copy()
    df1['drug_negation'] = df1['llm_output'].apply(lambda x: x['step_3_drug_exposure'] if x!='parse_failed' else x)
    df2 = dedup_parse(p2)
    df2 = df2[df2['report_id']!='report_id'].copy()
    df2['symptom_negation'] = df2['llm_output'].apply(lambda x: x['step2_analysis']['label'] if x!='parse_failed' else x)
    df2['symptom_negation'] = df2['symptom_negation'].apply(lambda x: 'N' if x=='NO CLEAR EVIDENCE OF SYMPTOM IN PATIENT' else 'Y')
    df3 = dedup_parse(p3)
    df3 = df3[df3['report_id']!='report_id'].copy()
    df3['cause_matches_input_drug'] = df3['llm_output'].apply(lambda x: x['etiology_classification'] if x!='parse_failed' else 'PARSING_ERROR')
    df3['cause_matches_input_drug'] = df3['cause_matches_input_drug'].apply(lambda x: 'Y' if x=='highly_suspect_drug_related' else ('N' if x=='highly_suspect_other_etiology' else 'O'))
    df4 = dedup_parse(p4)
    df4 = df4[df4['report_id']!='report_id'].copy()
    
    df4['steroid_for_symptom'] = df4['llm_output'].apply(lambda x: x['steroid_for_symptom'] if x!='parse_failed' else 'parse_failed')
    df4['immunotherapy_hold_for_symptom'] = df4['llm_output'].apply(lambda x: x['immunotherapy_hold_for_symptom'] if x!='parse_failed' else 'parse_failed')
    df4['intervene_target_symtom'] = df4.apply(lambda x: 'N' if x['steroid_for_symptom']==False and x['immunotherapy_hold_for_symptom']==False else 'Y', axis=1)
    df5 = dedup_parse(p5)
    df5 = df5[df5['report_id']!='report_id'].copy()
    
    df5['normalized_start_date_drug'] = df5['llm_output'].apply(lambda x: x['normalized_start_date'] if x!='parse_failed' else 'parse_failed')
    
    df6 = dedup_parse(p6)
    df6 = df6[df6['report_id']!='report_id'].copy()
    
    df6['normalized_start_date_symptom'] = df6['llm_output'].apply(lambda x: x['normalized_time'] if x!='parse_failed' else 'parse_failed')
    
    df67 = df6[['report_id', 'standard_symptom', 'normalized_start_date_symptom']].merge(df5[['report_id', 'standard_drug', 'normalized_start_date_drug']], on=['report_id'], how='left')

    df67['symptom_start_date_earlier_than_drug'] = df67.apply(symptom_earlier_than_drug, axis=1)

    modeldf3 = df3[['report_id', 'standard_drug', 'standard_symptom', 'cause_matches_input_drug']].copy()
    modeldf2 = df2[['report_id', 'standard_symptom', 'symptom_negation']].copy()
    modeldf4 = df4[['report_id', 'standard_symptom', 'intervene_target_symtom']].copy()
    modeldf1 = df1.groupby(['report_id', 'standard_drug'], as_index=False).agg({'drug_negation':lambda x: 'Y' if "Confirmed" in x.values else 'N'})
    modeldf67 = df67[['report_id', 'standard_drug', 'standard_symptom','normalized_start_date_symptom', 'normalized_start_date_drug','symptom_start_date_earlier_than_drug']].copy()

    result = modeldf3.merge(modeldf2, on = ['report_id',  'standard_symptom'], how = 'left')
    result = result.merge(modeldf4, on=['report_id', 'standard_symptom'], how='left')
    result = result.merge(modeldf1, on=['report_id', 'standard_drug'], how='left')
    result = result.merge(modeldf67, on=['report_id', 'standard_drug','standard_symptom'], how='left')
    results = select_priority_combination(
        df=result,
        group_cols = ["report_id", "standard_drug", "standard_symptom"],
        combo_cols = ["drug_negation", "symptom_negation", "cause_matches_input_drug", "intervene_target_symtom"],
        priority_order = info_order,
        drop_temp_cols=True,
        unknown_strategy="error"  # "error" | "last"
    )[['report_id', 'standard_drug', 'standard_symptom', "drug_negation", "symptom_negation", "cause_matches_input_drug", "intervene_target_symtom", 'normalized_start_date_symptom', 'normalized_start_date_drug',"symptom_start_date_earlier_than_drug"]]
    results['pred'] = results.apply(lambda x: ec(x), axis=1)
    results.to_csv('LLM_df.csv', index=False)


if __name__ == "__main__":
    main()
























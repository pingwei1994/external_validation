import pandas as pd
import re
import string
from collections import defaultdict
from tqdm import tqdm

def extract_ici_mentions(df: pd.DataFrame, ici_dict) -> pd.DataFrame:

    clean_list = [item for sublist in ici_dict.values() for item in sublist]
    clean_list = [word for word in clean_list if len(word) >= 3]

    def clean_variants(ici_dict):
        cleaned_dict = {}
        for ici_name, variants in ici_dict.items():
            seen = {}
            for var in variants:
                var_std = var.lower().rstrip(string.punctuation)
                if var_std not in seen:
                    seen[var_std] = var
            cleaned_dict[ici_name] = list(seen.values())
        return cleaned_dict

    # 清洗 variants
    ici_dict = clean_variants(ici_dict)

    # 换行符替换
    df = df.copy()
    # df['notes'] = df['notes'].str.replace('\n', '$%')

    # 存储匹配结果
    results = []

    for _, row in df.iterrows():
        report_id = row['report_id']
        note = row['notes'].lower()
        for ici_name, variants in ici_dict.items():
            for variant in variants:
                pattern = re.compile(re.escape(variant.lower()))
                for match in pattern.finditer(note):
                    results.append({
                        'report_id': report_id,
                        'notes': row['notes'],
                        'standard_drug': ici_name,
                        'matched_variant': variant,
                        'start_index': match.start()
                    })

    matches_df = pd.DataFrame(results)

    # 筛选每个位置最长的匹配
    matches_df['variant_length'] = matches_df['matched_variant'].str.len()
    matches_df = matches_df.merge(
        matches_df.groupby(['report_id', 'start_index'], as_index=False)
        .agg({'variant_length': 'max'})
        .rename(columns={'variant_length': 'max_length_variant'}),
        on=['report_id', 'start_index'],
        how='left'
    )
    matches_df = matches_df[matches_df['variant_length'] == matches_df['max_length_variant']]

    # 抽取完整词
    def extract_word_from_index(x):
        text = x['notes']
        start_idx = x['start_index']
        left = start_idx
        while left > 0 and text[left - 1].isalnum():
            left -= 1
        right = start_idx
        while right < len(text) and text[right].isalnum():
            right += 1
        return text[left:right]

    matches_df['theword'] = matches_df.apply(lambda x: extract_word_from_index(x), axis=1)

    # clean_list = ['nivolumab', 'Nivolumab', 'NIVOLUMAB', 'Nivolulmab','Nivo', 'OPDIVO','pembrolizumab', 'pembro',  'ipiliumumab', 'ipilimumab',  'pembrolizuamb', 'ipi','ipilumumab','PEMBROLIZUMAB',  'Keytruda', 'nivo',  'ipilumimab', 'ipilumuad', 'ipilumab',  'nivi', 'nivolomab',
    #    'nivolimab',  'Nivolumuab', 'Opdivo','C8D15Nivo', 'opdivo','Opidivo', 'IPILIMUMAB', 'Ipilimumab','Ipilmumab', 'hisipilimuamb', 'ipilubumab', 'Atezolizumab', 'nivolimumab', 'Ipi',  'novi','nivo12','Ipilumumab','nivolumabvs','atezolizumab', 'tremelimumab',
    #    'NivoluMab', 'NivoluMAb', 'Pembrolizumab','Pembro', 'YERVOY', 'ipilimuab', 'Yervoy', 'nivoumab', 'ipilmumuab', 'ipilimumad','yervoy', 'tecentriq', 'TECENTRIQ','ipillimumab','Novilumab', 'keytruda', 'KEYTRUDA']


    # matches_df = matches_df[matches_df['theword'].isin(clean_list)].reset_index(drop=True)
    matches_df = matches_df[
    ~((matches_df['matched_variant'].str.len() <= 4) & (~matches_df['theword'].isin(clean_list)))
].reset_index(drop=True)

    matches_df['drug_adjacent'] = matches_df.apply(
    lambda x: 
        x['notes'][max(0, x['start_index'] - 500): x['start_index']] + 
        '<drug> ' + x['matched_variant'] + ' </drug>' + 
        x['notes'][x['start_index'] + x['variant_length'] : x['start_index'] + x['variant_length'] + 500],
    axis=1
)
    # matches_df['drug_sentence_mask'] = matches_df.apply(lambda x: x['notes'][:x['drug_loc0']].split('$%')[-1] + '{drug_mask}' +x['notes'][x['drug_loc1']:].split('$%')[0] , axis=1)

    return matches_df


def extract_symptom_mentions(df: pd.DataFrame, symptom_dict) -> pd.DataFrame:

    clean_list = [item for sublist in symptom_dict.values() for item in sublist]
    
    def clean_variants(symptom_dict):
        cleaned_dict = {}
        for symptom_name, variants in symptom_dict.items():
            seen = {}
            for var in variants:
                var_std = var.lower().rstrip(string.punctuation)
                if var_std not in seen:
                    seen[var_std] = var
            cleaned_dict[symptom_name] = list(seen.values())
        return cleaned_dict

    # 清洗 variants
    symptom_dict = clean_variants(symptom_dict)

    # 换行符替换
    df = df.copy()
    # df['notes'] = df['notes'].str.replace('\n', '$%')

    # # 存储匹配结果
    # results = []

    # for _, row in df.iterrows():
    #     report_id = row['report_id']
    #     note = row['notes'].lower()
    #     for symptom_name, variants in symptom_dict.items():
    #         for variant in variants:
    #             pattern = re.compile(re.escape(variant.lower()))
    #             for match in pattern.finditer(note):
    #                 results.append({
    #                     'report_id': report_id,
    #                     'notes': row['notes'],
    #                     'standard_symptom': symptom_name,
    #                     'matched_variant': variant,
    #                     'start_index': match.start()
    #                 })

    # matches_df = pd.DataFrame(results)

    # # 筛选每个位置最长的匹配
    # matches_df['variant_length'] = matches_df['matched_variant'].str.len()
    # matches_df = matches_df.merge(
    #     matches_df.groupby(['report_id', 'start_index'], as_index=False)
    #     .agg({'variant_length': 'max'})
    #     .rename(columns={'variant_length': 'max_length_variant'}),
    #     on=['report_id', 'start_index'],
    #     how='left'
    # )
    # matches_df = matches_df[matches_df['variant_length'] == matches_df['max_length_variant']]

    results = []

    # 构造 (symptom_name, variant) 列表，并按长度降序
    all_variants = []
    for symptom_name, variants in symptom_dict.items():
        for variant in variants:
            all_variants.append((symptom_name, variant))
    all_variants.sort(key=lambda x: len(x[1]), reverse=True)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        report_id = row['report_id']
        note = row['notes'].lower()
    
        used_spans = []  # 记录已匹配区间 (start, end)
    
        for symptom_name, variant in all_variants:
            pattern = re.compile(re.escape(variant.lower()))
            for match in pattern.finditer(note):
                span = (match.start(), match.end())
    
                # 判断当前 span 是否与已有 span 重叠
                overlap = False
                for used_start, used_end in used_spans:
                    # 有重叠就跳过
                    if not (span[1] <= used_start or span[0] >= used_end):
                        overlap = True
                        break
    
                if not overlap:
                    used_spans.append(span)
                    results.append({
                        'report_id': report_id,
                        'notes': row['notes'],
                        'standard_symptom': symptom_name,
                        'matched_variant': variant,
                        'start_index': span[0]
                    })
    
    matches_df = pd.DataFrame(results)
    matches_df['variant_length'] = matches_df['matched_variant'].str.len()

    # 抽取完整词
    def extract_word_from_index(x):
        text = x['notes']
        start_idx = x['start_index']
        left = start_idx
        while left > 0 and text[left - 1].isalnum():
            left -= 1
        right = start_idx
        while right < len(text) and text[right].isalnum():
            right += 1
        return text[left:right]

    matches_df['theword'] = matches_df.apply(lambda x: extract_word_from_index(x), axis=1)

    # matches_df = matches_df[matches_df['theword'].isin(clean_list)].reset_index(drop=True)
    matches_df = matches_df[
    ~((matches_df['matched_variant'].str.len() <= 4) & (~matches_df['theword'].isin(clean_list)))
].reset_index(drop=True)

    matches_df['symptom_adjacent'] = matches_df.apply(
    lambda x: 
        x['notes'][max(0, x['start_index'] - 500): x['start_index']] + 
        '<symptom> ' + x['matched_variant'] + ' </symptom>' + 
        x['notes'][x['start_index'] + x['variant_length'] : x['start_index'] + x['variant_length'] + 500],
    axis=1
)

    return matches_df

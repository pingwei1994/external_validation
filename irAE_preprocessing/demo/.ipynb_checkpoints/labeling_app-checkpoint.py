import pandas as pd
import streamlit as st
import openpyxl
import os
import json
import sys
sys.path.append('../src')
import mention_extraction.drug_ici_config


st.set_page_config(layout="wide")

# Load data
symptom_path = '../data/symptom_mentions.csv'
drug_path = '../data/drug_mentions.csv'
symp_men = pd.read_csv(symptom_path)
drug_men = pd.read_csv(drug_path)

config_file_json = "drug_symptom_dicts.json"
def load_config(config_file_json):
    with open(config_file_json, 'r') as f:
        config = json.load(f)
    return config
config = load_config(config_file_json)
notes = pd.read_csv(f'../data/{config['ehr_file_name']}')

# notes_path = '../data/mock_ehr.csv'
# notes = pd.read_csv(notes_path)
symp_men = symp_men.merge(notes, on='report_id', how='left')
drug_men = drug_men.merge(notes, on='report_id', how='left')
# df['notes'] = df['notes'].apply(lambda x: x.replace('\n', '  '))
symp_men['end_index'] = symp_men['start_index'] + symp_men['variant_length']
drug_men['end_index'] = drug_men['start_index'] + drug_men['variant_length']
symp_men['real_variant'] = symp_men.apply(lambda x: x['notes'][x['start_index']:x['end_index']], axis=1)
drug_men['real_variant'] = drug_men.apply(lambda x: x['notes'][x['start_index']:x['end_index']], axis=1)
# Group data by patient first
s_grouped_by_patient = symp_men.groupby("patient_index")
d_grouped_by_patient = drug_men.groupby("patient_index")

# Initialize session state for tracking
if "current_patient_index" not in st.session_state:
    st.session_state.current_patient_index = None
if "current_report_id" not in st.session_state:
    st.session_state.current_report_id = None
if "current_drug" not in st.session_state:
    st.session_state.current_drug = None
if "current_symptom" not in st.session_state:
    st.session_state.current_symptom = None
if "labeled_data" not in st.session_state:
    st.session_state.labeled_data = {}
if "ehr_summary" not in st.session_state:
    st.session_state.ehr_summary = ""

# Create three columns
col1, col2, col3, col4 = st.columns([1, 5, 2, 2])

# Left Column: Patient selection, report details, EHR summary
with col1:
    st.subheader("Patient & Report Selection")
    patient_options = list(set(symp_men['patient_index'].unique().tolist() + drug_men['patient_index'].unique().tolist()))
    selected_patient_index = st.selectbox("Select Patient Index", patient_options)
    if selected_patient_index != st.session_state.current_patient_index:
        st.session_state.current_patient_index = selected_patient_index
        st.session_state.current_report_id = None
        st.session_state.current_drug = None
        st.session_state.current_symptom = None
        st.session_state.ehr_summary = ""

    patient_reports = list(set(s_grouped_by_patient.get_group(selected_patient_index)['report_id'].unique().tolist()+d_grouped_by_patient.get_group(selected_patient_index)['report_id'].unique().tolist()))
    selected_report_id = st.selectbox("Select Report ID", patient_reports)
    if selected_report_id != st.session_state.current_report_id:
        st.session_state.current_report_id = selected_report_id
        st.session_state.current_drug = None
        st.session_state.current_symptom = None
        st.session_state.ehr_summary = ""

    s_grouped_report = symp_men[symp_men['report_id'] == selected_report_id]
    d_grouped_report = drug_men[drug_men['report_id'] == selected_report_id]

with col2:
    # --- 分组 ---
    grouped_by_drug = d_grouped_report.groupby("standard_drug")
    grouped_by_symptom = s_grouped_report.groupby("standard_symptom")
    
    drug_list = list(grouped_by_drug.groups.keys())
    symptom_list = list(grouped_by_symptom.groups.keys())
    
    # --- 初始化 session state ---
    if st.session_state.current_drug is None:
        st.session_state.current_drug = drug_list[0]
    if st.session_state.current_symptom is None:
        st.session_state.current_symptom = symptom_list[0]
    
    # --- Drug 切换按钮 ---
    selected_drug = st.radio(
        "Select a drug:",
        options=drug_list,
        index=drug_list.index(st.session_state.current_drug),
        key="drug_radio", horizontal=True
    )
    if selected_drug != st.session_state.current_drug:
        st.session_state.current_drug = selected_drug
        st.rerun()
    
    # --- Symptom 下拉选择 ---
    selected_symptom = st.selectbox(
        "Select a symptom:",
        options=symptom_list,
        index=symptom_list.index(st.session_state.current_symptom),
        key="symptom_selectbox",
    )
    # 更新 session state
    if selected_symptom != st.session_state.current_symptom:
        st.session_state.current_symptom = selected_symptom
        st.rerun()
    
    # --- 获取当前 drug/symptom 的 group ---
    current_drug = st.session_state.current_drug
    current_symptom = st.session_state.current_symptom
    
    dgroup = grouped_by_drug.get_group(current_drug)
    sgroup = grouped_by_symptom.get_group(current_symptom)
    
    # 保证两个 group 的 notes 是相同的（可以添加 assert 检查）
    text = dgroup["notes"].iloc[0]
    assert text == sgroup["notes"].iloc[0], "Drug 和 Symptom 的 note 不一致"
    
    # 准备 variant 列表
    d_v_list = dgroup["real_variant"].dropna().tolist()
    d_v_locations = list(zip(dgroup["start_index"].dropna().astype(int), dgroup["end_index"].dropna().astype(int)))
    sorted_drugs = [(v, loc, '#b0e0e6') for v, loc in zip(d_v_list, d_v_locations)]
    
    s_v_list = sgroup["real_variant"].dropna().tolist()
    s_v_locations = list(zip(sgroup["start_index"].dropna().astype(int), sgroup["end_index"].dropna().astype(int)))
    sorted_ades = [(v, loc, 'yellow') for v, loc in zip(s_v_list, s_v_locations)]
    
    # 合并并按 start_index 排序
    all_highlights = sorted(sorted_drugs + sorted_ades, key=lambda x: x[1][0])
    
    # 高亮函数
    def highlight_variants(text, sorted_variants):
        text_chars = list(text)
        offset = 0
        for v, (start, end), color in sorted_variants:
            adjusted_start = start + offset
            adjusted_end = end + offset
            highlight_tag = f"<mark style='background-color: {color}' title='{v}'>{v}</mark>"
            text_chars[adjusted_start:adjusted_end] = list(highlight_tag)
            offset += len(highlight_tag) - (end - start)
        return "".join(text_chars)
    
    # 应用高亮
    highlighted_text = highlight_variants(text, all_highlights)
    
    # 展示
    st.subheader(f"Report ID: {selected_report_id}, Drug: {current_drug}, Symptom: {current_symptom}")
    st.markdown(f"<p style='font-size:16px'>{highlighted_text}</p>", unsafe_allow_html=True)

with col3:
    if current_symptom is not None:
        selected_ade = current_symptom
        st.write(f"### Label ADE: {selected_ade}")
        # --- Tier 1 Decision ---
        st.subheader("Tier 1: Decisive Rules")
        tier1_decision = st.radio("Does this symptom meet any Tier 1 rule?", [
            "Physician clearly confirms irAE",
            "Physician clearly negates irAE or confirms alternative cause",
            "Timing clearly excludes irAE",
            "Symptom Negation / Not a Symptom",
            "Drug Negation",
            "Baseline Symptoms for irAE Management",
            "ADE Note: LEGEND: 1 = Negative, 23 = Vague, 45 = positive",
            "No Decision was Made, Go to Scoring System"
        ], key=f"tier1_{selected_report_id}_{current_drug}_{current_symptom}")
    
        tier1_sentence = st.text_area(
                f"Input sentence supporting your judgment for Tier1",
                key=f"sentence_tier1_{selected_report_id}_{current_drug}_{current_symptom}"
            )
        # --- Tier 2 Scoring System ---
        st.subheader("Tier 2: Multi-Dimensional Evidence Scoring")
        scoring_fields = []
        dimensions = [
            ("Physician language / tone", "e.g., 'possibly immune-mediated'"),
            ("Symptom timing vs ICI", "e.g., symptom appears after ICI starts"),
            ("Management/Treatment direction", "e.g., steroid, ICI hold"),
            ("Response to ICI hold or steroid", "e.g., symptom improved"),
            ("Alternative explanation offered", "e.g., infection, cancer"),
        ]
    
        score_map = {"Tier1(NA)": -100, "Support (+1)": 1, "Neutral (0)": 0, "Against (-1)": -1}
        score_values = []
        evidence_sentences = []

        for i, (title, hint) in enumerate(dimensions):
            if i<2:
                st.markdown(f"**{i+1}. {title}**")
                score = st.radio(
                    f"Does this dimension support irAE? ({hint})",
                    list(score_map.keys()),
                    horizontal=True,
                    key=f"score_{i}_{selected_report_id}_{current_drug}_{current_symptom}"
                )
                sentence = st.text_area(
                    f"Input sentence supporting your judgment for: {title}",
                    key=f"sentence_{i}_{selected_report_id}_{current_drug}_{current_symptom}",
                    height=10
                )
                score_values.append(score_map[score])
                evidence_sentences.append(sentence)
            else:
                with col4:
                    st.markdown(f"**{i+1}. {title}**")
                    score = st.radio(
                        f"Does this dimension support irAE? ({hint})",
                        list(score_map.keys()),
                        horizontal=True,
                        key=f"score_{i}_{selected_report_id}_{current_drug}_{current_symptom}"
                    )
                    sentence = st.text_area(
                        f"Input sentence supporting your judgment for: {title}",
                        key=f"sentence_{i}_{selected_report_id}_{current_drug}_{current_symptom}",
                        height=10
                    )
                    score_values.append(score_map[score])
                    evidence_sentences.append(sentence)
        with col4:
            # --- Final Label ---
            st.subheader("Final Annotation")
            final_label = st.radio("Select Final Label", ["Positive", "Negative", "Vague"],
                                   key=f"final_label_{selected_report_id}_{current_drug}_{current_symptom}")

            # Submit scoring
            if st.button("Submit Score"):
                total_score = sum(score_values)
                st.success(f"Total Score: {total_score}")
                st.session_state[f"final_score_{selected_report_id}_{current_drug}_{current_symptom}"] = total_score
                
            def load_or_create_csv(csv_path):
                if os.path.exists(csv_path):
                    return pd.read_csv(csv_path)
                else:
                    return pd.DataFrame(columns=[
                        "report_id", "standard_drug", "standard_symptom",
                        "tier1_decision", "tier1_sentence", "final_label",
                        "score_total", "score_dimensions", "score_sentences"
                    ])
            
            def save_label_to_csv(csv_path, label_row):
                df = load_or_create_csv(csv_path)
            
                # 查重条件：report_id + drug + symptom 都相同
                duplicate_mask = (
                    (df["report_id"] == label_row["report_id"]) &
                    (df["standard_drug"] == label_row["standard_drug"]) &
                    (df["standard_symptom"] == label_row["standard_symptom"])
                )
            
                # 如果已存在，先删掉旧的
                df = df[~duplicate_mask]
            
                # 添加新记录
                df = pd.concat([df, pd.DataFrame([label_row])], ignore_index=True)
            
                # 保存
                df.to_csv(csv_path, index=False)
            if st.button("Submit Label"):
                label_row = {
                    "report_id": selected_report_id,
                    "standard_drug": current_drug,
                    "standard_symptom": current_symptom,
                    "tier1_decision": tier1_decision,
                    "tier1_sentence": tier1_sentence,
                    "final_label": final_label,
                    "score_total": st.session_state.get(f"final_score_{selected_report_id}_{current_drug}_{current_symptom}", None),
                    "score_dimensions": score_values,
                    "score_sentences": evidence_sentences
                }
            
                csv_path = "../data/labeled_data/all_annotations.csv"
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
                save_label_to_csv(csv_path, label_row)
            
                st.success("✅ Annotation saved to CSV (with deduplication)")


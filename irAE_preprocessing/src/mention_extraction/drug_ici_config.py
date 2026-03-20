import json


ehr_file_name = 'mock_ehr.csv'

drug_dict = {
        'nivolumab':['NI','NIVOLUMAB','Nivo','Nivolumab','nivo','nivo,','nivolimab','nivolomab','nivolumab','nivoumab','novi','OPDIVO','Opdivo','Opidivo','opdivo','NI.','NIVOLUMAB.','Nivo.','Nivolumab.','nivo.','nivolimab.','nivolomab.','nivolumab.','nivoumab.','novi.','OPDIVO.','Opdivo.','Opidivo.','opdivo.','NivoluMAb', 'NivoluMab', 'Nivolulmab','Nivolumuab', 'Novilumab', 'nivi','nivolimumab', 'nivolumabvs' ],
        'ipilimumab': ['IPI','IPILIMUMAB','Ipi','Ipilimumab','ipi','ipilimuamb','ipilimumab','ipilimumad','ipiliumumab','ipillimumab','ipilmumuab','ipilubumab','ipilumab','ipilumimab','ipilumuad','ipilumumab','YERVOY', 'Yervoy','yervoy','ipilimuab','IPI.','IPILIMUMAB.','Ipi.','Ipilimumab.','ipi.','ipilimuamb.','ipilimumab.','ipilimumad.','ipiliumumab.','ipillimumab.','ipilmumuab.','ipilubumab.','ipilumab.','ipilumimab.','ipilumuad.','ipilumumab.','YERVOY.', 'Yervoy.','yervoy.','ipilimuab.','Ipilmumab', 'Ipilumumab'],
        'pembrolizumab': ['PEMBROLIZUMAB','Pembro','Pembrolizumab','pembro','pembro/','pembrolizuamb','pembrolizumab','KEYTRUDA','Keytruda','keytruda','PEMBROLIZUMAB.','Pembro.','Pembrolizumab.','pembro.','pembrolizuamb.','pembrolizumab.','KEYTRUDA.','Keytruda.','keytruda.'],
        'Atezolizumab': ['Atezolizumab', 'atezolizumab', 'TECENTRIQ', 'tecentriq','Atezolizumab.', 'atezolizumab.', 'TECENTRIQ.', 'tecentriq.'],
        'tremelimumab': ['tremelimumab', 'tremelimumab.']
    }


symptom_dict = {
    "Arthritis": ['Arthritis', 'arthritis', 'arthiritis', 'ra'],
    "Goiter": ['Goiter', 'goiter'],
    "Hypertension": ['Hypertension', 'hypertension', 'HTN', 'htn', 'HYPERTENSION'],
    "Hypothyroid": ['Hypothyroid', 'hypothyroid', 'hypothyroidism'],
    "Rash": ['Rash', 'rash'],
    "Edema": ['edema', 'Edema', 'Swelling', 'swelling'],
    "Erythema": ['erythema', 'Erythema', 'Redness', 'redness'],
    "Increased TSH": ['increased TSH', 'Elevated TSH'],
    "Pruritus": ['prurit', 'Prurit', 'pruritis', 'pruritus','itch', 'itching', 'itchy'],
    "Abdominal Pain": ['abdominal pain', 'Abdominal pain', 'abd pain', 'abdominal cramp'],
    "Diarrhea": ['diarrhea', 'Diarrhea', 'diarhea', 'loose stool', 'Loose stool', 'watery stool', 'frequent stool', 'Frequent stool', 'Frequent BM\'s', 'frequent bowel movement'],
    "Shortness of Breath": ['short of breath', 'shortness of breath', 'Shortness of breath','Shortness of Breath','Shortness', 'shortness', 'SOB', 'sob', 'dyspnea', 'Dyspnea', 'dypsnea'],
    "Arthralgia": ['Arthralgia', 'arthralgia', 'joint pain', 'Joint pain', 'Joint Pain', 'joint ache', 'joint swelling', 'athralgias'],
    "Cough": ['Cough', 'cough'],
    "Dermatitis": ['Dermatitis', 'dermatitis'],
    "Dry Skin": ['Dry skin', 'skin dryness'],
    "Fatigue": ['Fatigue', 'fatigue', 'FATIGUE'],
    "Myalgia": ['Myalgia', 'myalgia', 'muscle pain', 'Muscle pain'],
    "Muscle Weakness": ['Muscle weakness', 'muscle weakness', 'Muscle Weakness'],
    "Stiffness": ['stiffness', 'Stiffness'],
    "Back Pain": ['back pain', 'Back pain', 'Back Pain'],
    "Osteoarthritis": ['osteoarthritis'],
    "Adrenal Insufficiency": ['Adrenal insufficiency', 'adrenal insufficiency'],
    "Low TSH": ['low TSH'],
    "DOE": ['DOE', 'doe', 'dyspneic with exertion', 'dyspnea with exertion', 'Dyspnea on exertion', 'dyspnea on exertion'],
    "Diabetes": ['DIABETES', 'Diabetes', 'diabetes'],
    "Rheumatoid": ['rheumatoid'],
    "Pneumonitis": ['pneumonitis', 'Pneumonitis', 'pnuemonitis'],
    "Appetite Decrease": ['appetite is decreasing', 'appetite has been low', 'decreased appetite', 'poor appetite', 'appetite is decreased', 'appetite has decreased', 'appetite: poor', 'appetite poor', 'lack of appetite', 'low appetite', 'loss of appetite'],
    "Numbness": ['numbness'],
    "Transaminitis": ['transaminitis', 'Transaminitis'],
    "Dizziness": ['Dizziness', 'dizziness', 'dizzy', 'lightheaded', 'Lightheaded'],
    "Chest Pain": ['Chest pain', 'chest pain', 'pleuritic chest pain', 'pain in upper chest'],
    "Syncope": [ 'Syncope', 'syncope', 'Near syncope', 'near syncopal episode', 'near-syncopal episode', 'syncopal episode', 'syncopal episodes', 'syncopal events', 'lost consciousness'],
    "Fibrosis": ['fibrosis', 'Fibrosis'],
    "Tachycardia": ['tachycardia', 'Tachycardia'],
    "Afib": ['Afib', 'afib', 'AFIB'],
    "Increased Oxygen Demand": ['Increased oxygen demands', 'increased oxygen'],
    "Hepatitis": ['hepatitis', 'Hepatitis', 'HEPATITIS', 'hepatis'],
    "Hyperthyroid": ['Hyperthyroid', 'hyperthyroid', 'hyperthyroidism'],
    "Blood in Stool": ['blood in stool', 'blood in his stool', 'BRBPR', 'bloody stool', 'Hematochezia', 'hematochezia', 'dark stool'],
    "Rheumatology": ['RHEUMATOLOGY', 'Rheumatology', 'rheumatology'],
    "Colitis": ['colitis', 'Colitis', 'colities', 'colities,'],
    "Thromboembolism": ['Thromboembolism', 'VTE'],
    "Arrhythmia": ['arrhythmia'],
    "CPK Elevation": ['ymptomatic elevation in CPK', 'symptomatic elevation in CPK (341)'],
    "Muscle Tenderness": ['muscle tenderness'],
    "Muscle Cramps": ['muscle cramps'],
    "Abnormal LFTs": ['Abnormal LFTs', 'abnormal LFTs', 'LFT abnormalities', 'LFT derangement', 'elevated liver enzyme', 'Elevated liver enzyme', 'Elevated LFTs', 'elevated LFTs', 'elevated lft', 'increased lft\'s', "increase lft's", "increased lft'", "elevation of the lft's", 'lfts continue to improve'],
    "Hepatotoxicity": ['hepatotoxicity', 'inflammatory component of liver injury'],
    "Encephalopathy": ['encephalopathy'],
    "Thyroiditis": ['thyroiditis'],
    "T4 Low": ['t4 low'],
    "Bowel Movements": ['bowel movements'],
    "Coagulopathy":['Coagulopathy', 'coagulopathy'],
}


# Define JSON data
drug_symptom_dicts = {
    'ehr_file_name' : ehr_file_name,
    'drug_dict' : drug_dict,
    'symptom_dict' : symptom_dict
}



# Specify file path
file_path = "./drug_symptom_dicts.json"


# Write JSON data to file
with open(file_path, 'w') as f:
    json.dump(drug_symptom_dicts, f)

    
print("JSON file created successfully.")
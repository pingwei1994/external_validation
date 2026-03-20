# External Validation Pipeline

A three-stage pipeline for processing EHR data through preprocessing, LLM inference, and evidence consolidation.

---

## Step 1: Preprocessing

### 1. Navigate to the preprocessing folder
```bash
cd external_validation/irAE_preprocessing
```

### 2. Prepare Your EHR Data

Place your EHR CSV file in the `data/` folder. The CSV **must** contain exactly three columns:

| Column | Description |
|---|---|
| `patient_index` | Unique patient identifier (similar to MRN) |
| `report_id` | Unique identifier for each EHR report **(primary key)** |
| `notes` | Text content of the EHR report |

> ⚠️ **Important:** Every row must have a unique `report_id`.

### 3. Update the Config File

Open `src/mention_extraction/drug_ici_config.py`, find the `ehr_file_name` variable, and set it to your EHR CSV filename:
```python
ehr_file_name = "mock_ehr.csv"
```

### 4. Run Preprocessing
```bash
cd demo
python preprocessing_demo.py
```

### 5. Verify Output

After completion, confirm that the following folder exists and contains `df1` through `df6`:
```
external_validation/irAE_preprocessing/llm_dfs/
```

---

## Step 2: LLM Inference

### 1. Navigate to the LLM inference folder
```bash
cd external_validation/LLM_inference
```

### 2. Configure Your Hugging Face Token

In `config_instruction.yaml`, replace the placeholder with your HF token:
```yaml
token: |-
  hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Install Dependencies

This pipeline was tested on **Python 3.12** with the following package versions:

| Package | Version |
|---|---|
| `huggingface-hub` | 0.30.2 |
| `pandas` | 2.3.0 |
| `PyYAML` | 6.0.1 |
| `torch` | 2.7.0 |
| `tqdm` | 4.66.4 |
| `transformers` | 4.51.3 |

### 4. Run LLM Inference
```bash
python LLM_infer.py
```

### 5. Verify Output

After completion, confirm that 6 result files appear in:
```
external_validation/LLM_inference/LLM_results/
```

---

## Step 3: Evidence Consolidation

### 1. Navigate to the consolidation folder
```bash
cd external_validation/Evidence_consolidation
```

### 2. Run Evidence Consolidation
```bash
python evidence_consolidation.py
```

### 3. Verify Output

After completion, confirm the output file exists at:
```
external_validation/Evidence_consolidation/LLM_df.csv
```


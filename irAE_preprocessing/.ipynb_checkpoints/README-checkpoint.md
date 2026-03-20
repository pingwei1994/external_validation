# irAE NLP Pipeline

This project consists of two main components: **Data Preprocessing** and an **Annotation App** for labeling.

---

## 📂 Project Structure
- **`src/`** — Core Python scripts for data preprocessing and mention extraction.  
- **`demo/`** — Example scripts for running preprocessing and the annotation app.  
- **`data/`** — Local storage for input EHR data *(not pushed to GitHub)*.

---

## 🚀 How to Run

### **1. Prepare Your EHR Data**
- Place your EHR CSV file in the `data/` folder.  
- The CSV **must** have exactly three columns:  

| Column         | Description |
|----------------|-------------|
| `patient_index` | Similar to MRN; each patient has a unique ID. |
| `report_id`     | Unique identifier for each EHR report (primary key). |
| `notes`         | Text content of the EHR report. |

⚠️ Ensure **`report_id`** is unique for every row.

---

### **2. Update the Config File**
Open `src/mention_extraction/drug_ici_config.py`  
Find the `ehr_file_name` variable and set it to your EHR CSV filename.

Example:
```python
ehr_file_name = "mock_ehr.csv"
```
---

### **3. Run Preprocessing**

```python
cd demo
python preprocessing_demo.py
```
---

### **4. Launch the Annotation App**

```python
cd demo
streamlit run labeling_app.py
```
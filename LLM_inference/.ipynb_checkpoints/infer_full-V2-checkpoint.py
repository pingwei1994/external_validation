import os
import math
import argparse
import pandas as pd
import torch
import re
from pathlib import Path
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def get_global_rank_world() -> tuple[int, int]:
    """Compute a global rank/world across jobs.

    Priority:
      1) SLURM job array: use SLURM_ARRAY_TASK_ID and SLURM_NTASKS
      2) Manual multi-job: use env N_JOBS and JOB_INDEX (you set at submit)
      3) Fallback: single job
    """
    local_rank  = int(os.getenv("SLURM_PROCID", "0"))
    local_world = int(os.getenv("SLURM_NTASKS", "1"))

    # Prefer SLURM array if present
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        job_index = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
        n_jobs = int(os.getenv("SLURM_ARRAY_TASK_COUNT", os.getenv("N_JOBS", "1")))
    else:
        # Manual multi-job mode (for independently submitted jobs)
        job_index = int(os.getenv("JOB_INDEX", "0"))
        n_jobs    = int(os.getenv("N_JOBS", "1"))

    global_rank  = job_index * local_world + local_rank
    global_world = n_jobs    * local_world
    return global_rank, global_world


def parse_args():
    ap = argparse.ArgumentParser(description="Minimal multi-worker LLM inference (single GPU per task).")
    ap.add_argument("--input_csv", required=True)        # e.g. "$TMPDIR/in/data.csv"
    ap.add_argument("--persist_dir", required=True)      # durable output dir
    ap.add_argument("--chunksize", type=int, default=10)
    ap.add_argument("--save_interval", required=True)
    ap.add_argument("--model-dir", type=str,
                    default="/fs/ess/PCON0020/Zhimo/medgemma-27b-fixed",
                    help="Local model path (or HF id if you have internet).")
    ap.add_argument("--load-in-4bit", action="store_true", default=True,
                    help="Use 4-bit quantization via bitsandbytes.")
    return ap.parse_args()


def load_model(model_dir: str, load_in_4bit: bool, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        quantization_config=qconf,
        device_map={"": 0},
    )

    model.eval()
    return tok, model

def format_0_shot_prompt(row):
    # return ('Given the title and summary of the following GSE data from GEO. Does the record satisfy both of the following: \
    #         1. Related to breast cancer. \
    #         2. We can reasonably deduce from the summary that it contains data obtained from bulkATAC-seq.'
    #         '### Title ###\n' + 
    #         row['Series_title'] + 
    #         '###Summary###\n' + 
    #         row['Series_summary'] +
    #         '###Design###\n' + 
    #         row['Series_overall_design'] +
    #         '### Answer: ###'
    #        )

    return (f'''Clinical Note:{row['drug_adjacent']},
    
                Input Drug: {row['standard_drug']}

                Please follow the steps.
                ''')

def flush(output_csv, accelerator, buffer, write_header):
    if not buffer:
        return
    pd.DataFrame(buffer).to_csv(
        output_csv,
        mode="a",
        header=write_header,
        index=False
    )
    write_header = False        # header only on first flush
    accelerator.print(f"Wrote{len(buffer)} rows")
    buffer.clear()


def main():
    tmp = os.environ["TMPDIR"] 
    input_dir = Path(tmp) / "in"

    job_id = os.getenv("SLURM_JOB_ID")
    array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    array_task_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")
    rank  = int(os.getenv("SLURM_PROCID", "0"))
    world = int(os.getenv("SLURM_NTASKS", "1"))
    node  = int(os.getenv("SLURM_NODEID", "0"))
    host  = os.getenv("SLURMD_NODENAME")
    
    os.environ.pop("ACCELERATE_USE_CPU", None)
    os.environ.pop("ACCELERATE_CPU", None)
    
    args = parse_args()

    g_rank, g_world = get_global_rank_world()
    print(f"[global_rank={g_rank} global_world={g_world}]")
    
    input_csv = Path(args.input_csv)
    persist = Path(args.persist_dir) / f"global-rank-{g_rank}"
    persist.mkdir(parents=True, exist_ok=True)
    chunk_size = args.chunksize
    
    accelerator = Accelerator(cpu=False)     # no DDP; just device + clean printing
    device = accelerator.device

    accelerator.print(f"[startup] device={device}, cuda_available={torch.cuda.is_available()}, "
                  f"n_gpus={torch.cuda.device_count()}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    accelerator.print(f"[startup] shard {rank}/{world} on {os.uname().nodename} device={device}")

    tok, model = load_model(args.model_dir, args.load_in_4bit, device)

    accelerator.print(f'Model loaded on node {node}.')

    # role_instruction = '''  
    #                      # Clinical Drug Analysis Assistant
                        
    #                     You are a clinical language assistant that analyzes clinical notes to determine if a specific drug was taken by a patient before the note was written.
                        
    #                     ## Input
    #                     - **Clinical Note**: A piece of clinical text containing a drug marked with `<drug>...</drug>` tags
    #                     - **Input Drug**: The specific drug name you need to analyze
                        
    #                     ## Task Overview
    #                     Follow these steps to analyze whether the tagged drug matches the input drug and if it was taken before the note was written.
                        
    #                     ---
                        
    #                     ## Step 1: Extract the Drug Object
    #                     Locate the exact word or phrase marked inside `<drug>...</drug>` tags. This is the only drug object you need to analyze. **Ignore all other drugs mentioned in the text.**
                        
    #                     ## Step 2: Summary the DRUG Object
    #                     Write a brief summary sentence about the drug object within the <drug>...</drug> tags, drawing exclusively from information contained in the **Clinical Note**—do not add external knowledge.
                        
    #                     ## Step 3: Decision Tree Analysis
    #                     Use the sentence from Step 2 and the drug object from Step 1 to answer the following questions:
                        
    #                     ### Question 1: Drug Match Verification
    #                     **Q1: Does the object marked inside `<drug>...</drug>` match the input drug?**
                        
    #                     Answer options: `[DRUG MATCH / NOT MATCH]`
                        
    #                     **Examples:**
    #                     - **NOT MATCH**: Clinical note contains "Patient has `<drug>`IPI`</drug>` score: 3", input drug is "ipilimumab" → The tagged "IPI" refers to a scoring system, not ipilimumab
    #                     - **DRUG MATCH**: Clinical note contains "Patient was on `<drug>`ipi`</drug>` + nivo", input drug is "ipilimumab" → The tagged "ipi" is an abbreviation for ipilimumab
                        
    #                     ### Question 2: Temporal Analysis
    #                     **Q2: Does the sentence from Step 2 indicate the drug was taken before the note was written?**
                        
    #                     Answer options: `[YES / NO]`
                        
    #                     **Answer YES if the sentence clearly indicates prior drug administration:**
    #                     1. **Cycle numbers mentioned** (e.g., "Cycle 2", "C3D2", "5 cycles completed")
    #                        - **Special attention for Cycle 1**: If Cycle 1 occurred on the day the note was written → answer NO. If Cycle 1 occurred before the note date → answer YES
    #                     2. **Past tense with dates** (e.g., "Drug started on [date]")
    #                     3. **Drug discontinuation/hold** (e.g., "Drug was held" or "discontinued") - can only hold/discontinue after administration
                        
    #                     **Answer NO if:**
    #                     1. Drug was not taken (e.g., "Patient not eligible for drug")
    #                     2. Drug administration is same-day as note
    #                     3. Drug is mentioned as future plan
    #                     4. Drug mentioned only in study context or consent
    #                     5. Drug mentioned only in prescription without confirmation of administration
                        
    #                     ---
                        
    #                     ## Decision Tree Rules
                        
    #                     1. **Start with Q1:**
    #                        - If answer is `[NOT MATCH]` → **Stop and return `[NOT DRUG]`**
    #                        - If answer is `[DRUG MATCH]` → **Proceed to Q2**
                        
    #                     2. **If proceeding to Q2:**
    #                        - If answer is `[YES]` → **Stop and return `[DRUG TAKEN]`**
    #                        - If answer is `[NO]` → **Stop and return `[DRUG NOT TAKEN]`**
                        
    #                     ---
                        
    #                     ## Output Format
                        
    #                     Provide your analysis in the following JSON format:
                        
    #                     ```json
    #                     {
    #                       "step1_extracted_drug_object": "string - exact text within <drug> tags",
    #                       "step2_summary_sentence": "string - the generated summary",
    #                       "step3_analysis": {
    #                         "q1_drug_match": {
    #                           "answer": "DRUG MATCH or NOT MATCH",
    #                           "reasoning": "string - explanation for the match determination"
    #                         },
    #                         "q2_temporal_analysis": {
    #                           "answer": "YES, NO, or N/A (if Q1 was NOT MATCH)",
    #                           "reasoning": "string - explanation for temporal determination or N/A if not applicable"
    #                         }
    #                       },
    #                       "final_result": "NOT DRUG, DRUG TAKEN, or DRUG NOT TAKEN",
    #                       "summary": "string - brief explanation of the final determination"
    #                     }
    #                     ```
                        
    #                     ## Example Output
                        
    #                     ```json
    #                     {
    #                       "step1_extracted_drug_object": "ipi",
    #                       "step2_complete_sentence": "Patient completed cycle 3 of ipi + nivo treatment last month.",
    #                       "step3_analysis": {
    #                         "q1_drug_match": {
    #                           "answer": "DRUG MATCH",
    #                           "reasoning": "'ipi' is a common abbreviation for ipilimumab"
    #                         },
    #                         "q2_temporal_analysis": {
    #                           "answer": "YES",
    #                           "reasoning": "Cycle 3 completion indicates prior administration, and 'last month' confirms it occurred before the note was written"
    #                         }
    #                       },
    #                       "final_result": "DRUG TAKEN",
    #                       "summary": "The tagged drug 'ipi' matches ipilimumab and the sentence indicates it was administered before the note was written (cycle 3 completed last month)."
    #                     }
    # '''
    role_instruction = '''

                    You are a clinical language assistant that analyzes clinical notes to determine if a patient was exposed to a specific drug.
                    
                    ## Input Parameters
                    - **Clinical Note**: Clinical text containing a drug marked with `<drug>...</drug>` tags
                    - **Input Drug**: The specific drug name to analyze for past exposure
                    
                    ## Analysis Instructions
                    
                    ### Step 1: Identify the Tagged Drug
                    Locate the exact word or phrase marked inside `<drug>...</drug>` tags in the clinical note. This is the **only** drug object you should analyze. **Ignore all other drugs mentioned in the text.**
                    
                    ### Step 2: Verify Drug Match
                    Determine if the tagged object refers to the input drug.
                    
                    **Decision Criteria:**
                    - **DRUG MATCH**: The tagged text is the input drug itself, or a valid abbreviation/alternative name for it
                    - **NOT MATCH**: The tagged text refers to something else (e.g., a scoring system, different drug, unrelated term)
                    
                    **Examples:**
                    - ❌ **NOT MATCH**: Note contains "Patient has `<drug>`IPI`</drug>` score: 3", input drug is "ipilimumab" → "IPI" refers to a scoring system
                    - ✅ **DRUG MATCH**: Note contains "Patient was on `<drug>`ipi`</drug>` + nivo", input drug is "ipilimumab" → "ipi" is an abbreviation for ipilimumab
                    
                    ### Step 3: Confirm Drug Exposure
                    Determine if the sentence confirms the patient was exposed to the drug.
                    
                    **Decision Criteria:**
                    - **Confirm**: The sentence clearly indicates the patient did receive the drug.
                    - **Denied**: The note explicitly states the patient did NOT receive or take the drug.
                    - **Vague** : The information is ambiguous or insufficient to determine drug exposure.

                    ### Step 4: Initiated today? (only if Step 3 = Confirmed)
                    If Step 3 = Confirmed, determine whether the note explicitly states the drug was initiated/started today (first dose given on the same day as the note).

                    **Decision Criteria:**
                    - **Yes** : clear phrasing such as “started X today”, “received first dose of X today”, “initiated X today”, or equivalent.
                    - **No** : explicit or implicit phrasing indicates prior administration not on the same day (e.g., “completed cycle 2”, “received last month”, “continues on X”).
                    - **Vague** : exposure confirmed but timing relative to the note is unclear (e.g., “patient is on X” without timing; “admitted for first dose” ambiguous).
                    - **Not Applicable** : If Step 3 ≠ Confirmed, set this field to N/A.
                    
                    ---
                    
                    ## Output Format
                    
                    Provide your analysis as a JSON object with the following structure:
                    
                    ```json
                     {
                      "tagged_drug_text": "<text inside <drug> tags>",
                      "input_drug": "<input drug>",
                      "step_2_drug_match": "DRUG MATCH | NOT MATCH",
                      "step_2_reasoning": "<brief>",
                      "step_3_drug_exposure": "Confirmed | Denied | Vague",
                      "step_3_reasoning": "<brief>",
                      "step_4_initiated_today": "YES | NO | VAGUE | Not Applicable",
                      "step_4_reasoning": "<brief>",
                      "final_conclusion": "<one-sentence summary>",
                      "relevant_sentence": "<the sentence used for decision>"
                    }

                    ```
                    
                    ### Example Output:
                    
                    ```json
                    {
                      "tagged_drug_text": "ipilimumab",
                      "input_drug": "ipilimumab",
                      "step_2_drug_match": "DRUG MATCH",
                      "step_2_reasoning": "Exact match",
                      "step_3_drug_exposure": "Confirmed",
                      "step_3_reasoning": "Note indicates administration occurred",
                      "step_4_initiated_today": "YES",
                      "step_4_reasoning": "Sentence states 'started ipilimumab today', explicitly first dose given today",
                      "final_conclusion": "Patient received ipilimumab and the first dose was initiated on the note date",
                      "relevant_sentence": "Patient started ipilimumab today."
                    }

                    ```
                    ---
                    
                    ## Important Notes
                    - Analyze **only** the drug marked with `<drug>...</drug>` tags
                    - Consider context carefully to distinguish between drug names and other medical terms
                    - Prescription alone does not confirm exposure
                    - Same-day exposure is not considered "past" exposure
                    - Always provide clear reasoning for your decisions


    '''
    system_instruction = f"SYSTEM INSTRUCTION: think silently if needed. {role_instruction}"
    max_new_tokens     = 1500

    # Everyone reads same file; only 1/world chunks get processed by this rank
    reader = pd.read_csv(
        input_csv,
        chunksize=chunk_size,
        dtype_backend="pyarrow",    # optional: memory-friendly
        low_memory=False            # consistent parsing
    )

    for i, df in enumerate(reader):
        if (i % g_world) != g_rank:
            continue  # not my chunk

        df['Prompt'] = df.apply(format_0_shot_prompt, axis = 1)

        if os.path.exists(persist / f"part_chunk{i:06d}.csv"):
            done = set(pd.read_csv(persist / f"part_chunk{i:06d}.csv", usecols=["row_idx"])["row_idx"])
            write_header = False
            accelerator.print(f"Resume: {len(done)} rows already scored")
        else:
            done, write_header = set(), True
        
        buffer = []                # first write gets header

        for idx, row in df.iterrows():
            print(row)
            if idx in done:
                continue
        
            messages = [
                {
                    "role": "system",
                    "content": system_instruction
                },
                {
                    "role": "user",
                    "content": row["Prompt"]
                }]
                
            inputs = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
        
            inp_len = inputs["input_ids"].shape[-1]
        
            with torch.inference_mode():
                out = model.generate(**inputs,
                                     max_new_tokens=max_new_tokens,
                                     do_sample=False)[0][inp_len:]
            raw = tok.decode(out, skip_special_tokens=True)
        
            # thought, response = ("", raw)
            # if "<unused95>" in raw:
            #     thought, response = raw.split("<unused95>", 1)
            #     thought = thought.replace("<unused94>thought\n", "")
            # m = re.search(r"===FINAL ANSWER===\s*([01])", response)
            # pred_label = str(int(m.group(1))) if m else ""

            # def parse_response(response_string):
            #     pattern = r'```json\s*(.*?)\s*```'
            #     match = re.search(pattern, response_string, re.DOTALL)
            #     try:
            #         if match:
            #             return json.loads(match.group(1).strip())
            #         return json.loads(response_string)
            #     except Exception:
            #         return None

            # json_response = parse_response(raw)
            # if json_response is None:
            #     json_response = {
            #         "error": "Failed to parse LLM response as JSON after two attempts",
            #         "raw_response": response
            #     }

            # accelerator.print(f"node {node} finished processing row {row['idx']}.")
            buffer.append(
                row.to_dict() | {
                    "row_idx":     idx,
                    "response":    raw.split('<unused95>')[-1],
                }
            )
            if len(buffer) >= int(args.save_interval):
                flush(persist / f"part_chunk{i:06d}.csv", accelerator, buffer, write_header)
        
        flush(persist / f"part_chunk{i:06d}.csv", accelerator, buffer, write_header)            # final write

if __name__ == "__main__":
    main()
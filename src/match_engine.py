import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import textwrap
import time
import os
import json
from pathlib import Path

# --- FIX: Stop Mac Threading Collisions ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- PANDAS SETTINGS ---
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / 'data' / 'processed' / 'onet_knowledge_base.csv'
INPUT_PATH = BASE_DIR / 'data' / 'raw' / 'user_survey_input.csv'
OUTPUT_PATH = BASE_DIR / 'data' / 'output' / 'matched_jobs_results.csv'

MODEL_NAME = 'all-MiniLM-L6-v2'

# --- CATEGORY WEIGHTS (Must add up to 1.0) ---
WEIGHT_TASKS = 0.60
WEIGHT_SKILLS = 0.20
WEIGHT_TITLES = 0.10
WEIGHT_TOOLS = 0.10

def encode_weighted_tasks(model, df):
    """
    Vector Weighting (The Gold Standard)
    Unpacks JSON tasks, encodes them individually, multiplies by their O*NET weight, 
    and returns a mathematically precise Average Task Vector for the occupation.
    """
    print(" -> Encoding Tasks (Applying Mathematical Importance Weights)...")
    all_kb_vectors = []
    
    for idx, row in df.iterrows():
        json_str = row['Structured_Tasks']
        
        # 1. Fallback if no tasks exist
        if not isinstance(json_str, str) or not json_str.strip():
            all_kb_vectors.append(torch.zeros(model.get_sentence_embedding_dimension()))
            continue
            
        try:
            tasks = json.loads(json_str)
        except Exception:
            all_kb_vectors.append(torch.zeros(model.get_sentence_embedding_dimension()))
            continue
            
        # 2. Extract texts and weights
        texts = [t.get('task', '') for t in tasks if t.get('task', '')]
        weights = [t.get('weight', 0.50) for t in tasks if t.get('task', '')]
        
        if not texts:
            all_kb_vectors.append(torch.zeros(model.get_sentence_embedding_dimension()))
            continue
            
        # 3. Vectorize all tasks for this job at once (Batch Encoding for speed)
        task_vecs = model.encode(texts, convert_to_tensor=True)
        
        # 4. THE MATRIX MATH: Multiply each task vector by its specific importance weight
        # We reshape the weights list into a tensor column so PyTorch can multiply it against the vector arrays
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(task_vecs.device).unsqueeze(1)
        weighted_vecs = task_vecs * weight_tensor
        
        # 5. Average the weighted vectors to create the Final Task Profile
        # We divide by the sum of the weights to normalize the mathematical magnitude
        avg_vector = torch.sum(weighted_vecs, dim=0) / torch.sum(weight_tensor)
        all_kb_vectors.append(avg_vector)
        
    return torch.stack(all_kb_vectors)

def encode_long_text(model, text, chunk_size=800):
    """
    The Chunking Engine (For Skills, Tools, and Titles)
    Bypasses the 256 token limit without losing data.
    """
    if not isinstance(text, str) or text.strip() == "":
        return torch.zeros(model.get_sentence_embedding_dimension())

    chunks = textwrap.wrap(text, width=chunk_size)
    if not chunks:
        return torch.zeros(model.get_sentence_embedding_dimension())
        
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    document_embedding = torch.mean(chunk_embeddings, dim=0)
    return document_embedding

def load_data():
    print(f"Loading Knowledge Base from {KB_PATH}...")
    kb_df = pd.read_csv(KB_PATH)
    kb_df.fillna("", inplace=True)

    # Auto-create dummy input if missing so you can test immediately
    if not INPUT_PATH.exists():
        print("[NOTICE] Creating mock user_survey_input.csv for testing...")
        os.makedirs(os.path.dirname(INPUT_PATH), exist_ok=True)
        dummy_data = pd.DataFrame({
            'User_ID': [1, 2],
            'User_Job_Title': ["Budget Director", "Code Writer"],
            'User_Job_Description': ["I manage finances, analyze budget reports, and handle investments.", 
                                     "I write python scripts, debug software, and manage databases."]
        })
        dummy_data.to_csv(INPUT_PATH, index=False)

    print(f"Loading User Input Data...")
    input_df = pd.read_csv(INPUT_PATH)
    input_df.fillna("", inplace=True)
    return kb_df, input_df

def run_matching_pipeline():
    start_time = time.time()
    kb_df, input_df = load_data()

    # loading the model and encoding the knowledge base

    print(f"\n[INIT] Loading AI Model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME, device='cpu')

    print("\n[PROCESSING] Vectorizing O*NET Database...")
    # 1. Encode Tasks using our new Mathematical Vector Weighting
    task_embeddings = encode_weighted_tasks(model, kb_df)
    
    # 2. Encode remaining categories using standard chunking
    print(" -> Encoding Skills...")
    skill_embeddings = torch.stack([encode_long_text(model, text) for text in kb_df['All_Tech_Skills']])
    print(" -> Encoding Tools...")
    tool_embeddings = torch.stack([encode_long_text(model, text) for text in kb_df['All_Tools']])
    print(" -> Encoding Alternate Titles...")
    title_embeddings = torch.stack([encode_long_text(model, text) for text in kb_df['All_Alt_Titles']])

    print("\n[PROCESSING] Vectorizing User Inputs...")
    user_texts = (input_df['User_Job_Title'] + " " + input_df['User_Job_Description']).tolist()
    query_embeddings = model.encode(user_texts, convert_to_tensor=True)

    print("\n[MATCHING] Calculating Multi-Vector Similarity...")
    # Calculate similarity scores (Outputs a 0.0 to 1.0 score for every category)
    task_scores = util.cos_sim(query_embeddings, task_embeddings)
    skill_scores = util.cos_sim(query_embeddings, skill_embeddings)
    tool_scores = util.cos_sim(query_embeddings, tool_embeddings)
    title_scores = util.cos_sim(query_embeddings, title_embeddings)

    # THE FINAL ALGORITHM: Apply the weights
    final_scores = (
        (task_scores * WEIGHT_TASKS) + 
        (skill_scores * WEIGHT_SKILLS) + 
        (title_scores * WEIGHT_TITLES) + 
        (tool_scores * WEIGHT_TOOLS)
    )

    # Extract Top 3 Results 
    match_1_codes, match_1_titles, match_1_scores = [], [], []
    match_2_codes, match_2_titles, match_2_scores = [], [], []
    match_3_codes, match_3_titles, match_3_scores = [], [], []

    # For each user, find the highest 3 scores in the final weighted matrix
    for i in range(len(input_df)):
        user_scores = final_scores[i]
        
        # PyTorch function to get the top K values and their original index positions
        top_3 = torch.topk(user_scores, k=3)
        top_3_scores = top_3.values.tolist()
        top_3_indices = top_3.indices.tolist()
        
        # --- Match 1 (Highest) ---
        idx_1 = top_3_indices[0]
        match_1_codes.append(kb_df.iloc[idx_1]['O*NET-SOC Code'])
        match_1_titles.append(kb_df.iloc[idx_1]['Title'])
        match_1_scores.append(round(top_3_scores[0] * 100, 2))
        
        # --- Match 2 ---
        idx_2 = top_3_indices[1]
        match_2_codes.append(kb_df.iloc[idx_2]['O*NET-SOC Code'])
        match_2_titles.append(kb_df.iloc[idx_2]['Title'])
        match_2_scores.append(round(top_3_scores[1] * 100, 2))

        # --- Match 3 ---
        idx_3 = top_3_indices[2]
        match_3_codes.append(kb_df.iloc[idx_3]['O*NET-SOC Code'])
        match_3_titles.append(kb_df.iloc[idx_3]['Title'])
        match_3_scores.append(round(top_3_scores[2] * 100, 2))

    # Save Results to the DataFrame as wide columns
    input_df['Match_1_SOC_Code'] = match_1_codes
    input_df['Match_1_Title'] = match_1_titles
    input_df['Match_1_Score'] = match_1_scores
    
    input_df['Match_2_SOC_Code'] = match_2_codes
    input_df['Match_2_Title'] = match_2_titles
    input_df['Match_2_Score'] = match_2_scores
    
    input_df['Match_3_SOC_Code'] = match_3_codes
    input_df['Match_3_Title'] = match_3_titles
    input_df['Match_3_Score'] = match_3_scores

    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    input_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n[SUCCESS] Pipeline finished in {time.time() - start_time:.2f}s")
    # print("\n--- SAMPLE PREDICTIONS (Top Match) ---")
    # print(input_df[['User_Job_Title', 'Match_1_Title', 'Match_1_Score']].head())

    print("\n--- SAMPLE PREDICTIONS (Top 3 Matches) ---")
    columns_to_show = [
        'User_Job_Title', 
        'Match_1_Title', 'Match_1_Score', 
        'Match_2_Title', 'Match_2_Score', 
        'Match_3_Title', 'Match_3_Score'
    ]
    print(input_df[columns_to_show].head())

if __name__ == "__main__":
    run_matching_pipeline()
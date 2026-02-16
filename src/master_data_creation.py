
import pandas as pd
import os
from pathlib import Path
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR         # Import the paths we defined in config.py

# --- PANDAS SETTINGS ---
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# --- CONFIGURATION ---
# These names must match the files inside data/raw/
FILES = {
    "occupation": "Occupation Data.txt",
    "tasks": "Task Statements.txt",
    "alt_titles": "Alternate Titles.txt",
    "tech_skills": "Technology Skills.txt",
    "tools": "Tools Used.txt"
}

def load_onet_file(filename, cols_to_keep):
    """
    Reads an O*NET .txt file from the RAW_DATA_DIR using Tab separator.
    """
    filepath = RAW_DATA_DIR / filename
    
    if not filepath.exists():
        print(f"[WARNING] File not found: {filepath}")
        return pd.DataFrame()

    print(f"Loading {filename}...")
    try:
        # O*NET text files use Tab separation (\t)
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        
        # Keep only the columns we need (if they exist)
        available_cols = [c for c in cols_to_keep if c in df.columns]
        return df[available_cols]
    except Exception as e:
        print(f"[ERROR] Could not read {filename}: {e}")
        return pd.DataFrame()

def flatten_data(df, group_col, text_col, new_col_name):
    """
    Groups many rows (like 20 tasks) into one long text string per job.
    """
    if df.empty:
        return pd.DataFrame(columns=[group_col, new_col_name])

    print(f"Flattening {text_col}...")
    df_flat = df.groupby(group_col)[text_col].apply(
        lambda x: ' | '.join(x.dropna().astype(str))
    ).reset_index()
    
    df_flat.rename(columns={text_col: new_col_name}, inplace=True)
    return df_flat

def make_master_profile(row):
    """
    Combines all job features into one searchable text block.
    """
    parts = [
        f"Job Title: {row.get('Title', '')}",
        f"Description: {row.get('Description', '')}",
        f"Alternate Titles: {row.get('All_Alt_Titles', '')}",
        f"Tasks: {row.get('All_Tasks', '')}",
        f"Tech Skills: {row.get('All_Tech_Skills', '')}",
        f"Tools: {row.get('All_Tools', '')}",
    ]
    return "\n".join(parts)

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # 1. Load the "Anchor" File
    df_main = load_onet_file(FILES["occupation"], ['O*NET-SOC Code', 'Title', 'Description'])

    # 2. Load and Flatten Auxiliary Files
    df_tasks = flatten_data(load_onet_file(FILES["tasks"], ['O*NET-SOC Code', 'Task']), 
                            'O*NET-SOC Code', 'Task', 'All_Tasks')
    
    df_alts = flatten_data(load_onet_file(FILES["alt_titles"], ['O*NET-SOC Code', 'Alternate Title']), 
                           'O*NET-SOC Code', 'Alternate Title', 'All_Alt_Titles')
    
    df_tech = flatten_data(load_onet_file(FILES["tech_skills"], ['O*NET-SOC Code', 'Example']), 
                           'O*NET-SOC Code', 'Example', 'All_Tech_Skills')
    
    df_tools = flatten_data(load_onet_file(FILES["tools"], ['O*NET-SOC Code', 'Example']), 
                           'O*NET-SOC Code', 'Example', 'All_Tools')

    # 3. Merge Everything
    print("Merging all datasets into Knowledge Base...")
    kb = df_main.merge(df_tasks, on='O*NET-SOC Code', how='left') \
               .merge(df_alts, on='O*NET-SOC Code', how='left') \
               .merge(df_tech, on='O*NET-SOC Code', how='left') \
               .merge(df_tools, on='O*NET-SOC Code', how='left')

    kb.fillna("", inplace=True)

    # 4. Create Master Profile
    print("Generating Master Profiles for AI processing...")
    kb['Master_Profile'] = kb.apply(make_master_profile, axis=1)

    # 5. Save to Processed folder
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "onet_knowledge_base.csv"
    
    kb.to_csv(output_path, index=False)

    print("-" * 30)
    print(f"SUCCESS! Knowledge base saved to: {output_path}")
    print(f"Total Occupations Processed: {len(kb)}")
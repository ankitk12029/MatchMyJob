
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
ONET_FILES = {
    "occupation": RAW_DATA_DIR / "Occupation Data.txt",
    "tasks_ratings": RAW_DATA_DIR / "Task Ratings.txt",
    "tasks": RAW_DATA_DIR / "Task Statements.txt",
    "alt_titles": RAW_DATA_DIR / "Alternate Titles.txt",
    "tech_skills": RAW_DATA_DIR / "Technology Skills.txt",
    "tools": RAW_DATA_DIR / "Tools Used.txt"
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
    df_main = load_onet_file(ONET_FILES["occupation"], ['O*NET-SOC Code', 'Title', 'Description'])

    # 2. Load and Flatten Auxiliary Files
    df_alts = flatten_data(load_onet_file(ONET_FILES["alt_titles"], ['O*NET-SOC Code', 'Alternate Title']), 
                           'O*NET-SOC Code', 'Alternate Title', 'All_Alt_Titles')
    
    df_tech = flatten_data(load_onet_file(ONET_FILES["tech_skills"], ['O*NET-SOC Code', 'Example']), 
                           'O*NET-SOC Code', 'Example', 'All_Tech_Skills')
    
    df_tools = flatten_data(load_onet_file(ONET_FILES["tools"], ['O*NET-SOC Code', 'Example']), 
                           'O*NET-SOC Code', 'Example', 'All_Tools')
    
    # Load BOTH files: The master task list and the un-filtered Task Ratings
    df_all_tasks = load_onet_file(ONET_FILES["tasks"], ['O*NET-SOC Code', 'Task ID', 'Task'])
    
    # Load raw ratings with the specific columns you requested
    df_ratings_raw = load_onet_file(ONET_FILES["tasks_ratings"], 
                                    ['O*NET-SOC Code', 'Title', 'Task ID', 'Task', 'Scale ID', 'Data Value'])

    df_ratings = pd.DataFrame()
    if not df_ratings_raw.empty:
        # Filter where Scale ID is 'IM' (Importance)
        df_ratings = df_ratings_raw[df_ratings_raw['Scale ID'] == 'IM'].copy()
        
        # Rename 'Data Value' to 'IMP_rating' so it flows perfectly into your existing code
        df_ratings.rename(columns={'Data Value': 'IMP_rating'}, inplace=True)
        
        # Keep only the columns needed for the merge to keep things clean
        df_ratings = df_ratings[['O*NET-SOC Code', 'Task ID', 'IMP_rating']]

    if not df_all_tasks.empty:
        # 2. LEFT JOIN: Keep all tasks, attach ratings if they exist
        if not df_ratings.empty:
            df_merged_task = pd.merge(df_all_tasks, df_ratings, on=['O*NET-SOC Code', 'Task ID'], how='left')
        else:
            # Fallback just in case ratings file is missing
            df_merged_task = df_all_tasks.copy()
            df_merged_task['IMP_rating'] = pd.NA

        # Conditionally append the importance rating
        def format_task(row):
            rating = row['IMP_rating']
            # Check if rating is missing (NaN, None, or empty string)
            if pd.isna(rating) or str(rating).strip() == '' or str(rating).lower() == 'nan':
                return str(row['Task'])
            else:
                # Append the rating, removing any decimal zeros (e.g., '88.0' -> '88')
                clean_rating = str(rating).replace('.0', '')
                return f"{row['Task']} [Importance: {clean_rating}]"

        df_merged_task['Task_Text'] = df_merged_task.apply(format_task, axis=1)
        
        # Flatten the combined text
        df_tasks = flatten_data(df_merged_task, 'O*NET-SOC Code', 'Task_Text', 'All_Tasks')
    else:
        df_tasks = pd.DataFrame(columns=['O*NET-SOC Code', 'All_Tasks'])


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
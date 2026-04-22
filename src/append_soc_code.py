import pandas as pd

# 1. Load the datasets
df_survey = pd.read_excel("/Users/ankitkatre/Library/CloudStorage/OneDrive-SanDiegoStateUniversity(SDSU.EDU)/Academics/matchmyjob/archive/Training Data Sets/soc_code_match.xlsx")
df_onet = pd.read_csv("/Users/ankitkatre/Library/CloudStorage/OneDrive-SanDiegoStateUniversity(SDSU.EDU)/Academics/matchmyjob/data/raw/Occupation Data.txt", sep='\t', dtype=str)

# 2. Fix the older taxonomy titles in the survey data to match modern O*NET
taxonomy_fixes = {
    'Software Developers, Applications': 'Software Developers',
    'Software Developers, Systems Software': 'Software Developers',
    'Accountants': 'Accountants and Auditors',
    'Financial Managers, Branch or Department': 'Financial Managers',
    'Stock Clerks, Sales Floor': 'Stockers and Order Fillers',
    'Secondary School Teachers, Except Special and Vocational Education': 'Secondary School Teachers, Except Special and Career/Technical Education',
    'Medical Secretaries': 'Medical Secretaries and Administrative Assistants',
    'Financial Analysts': 'Financial and Investment Analysts',
    'Software Quality Assurance Engineers and Testers': 'Software Quality Assurance Analysts and Testers',
    'Legal Secretaries': 'Legal Secretaries and Administrative Assistants'
}

# Replace the old titles with the updated ones
df_survey['fixed_title'] = df_survey['FINAL TITLE'].replace(taxonomy_fixes)

# 3. Clean strings to guarantee a match (lowercase and remove trailing spaces)
df_survey['match_title'] = df_survey['fixed_title'].astype(str).str.strip().str.lower()
df_onet['match_title'] = df_onet['Title'].astype(str).str.strip().str.lower()

# 4. Merge to fetch the SOC Code
df_merged = pd.merge(df_survey, df_onet[['O*NET-SOC Code', 'match_title']], on='match_title', how='left')

# Drop the temporary matching columns to keep your data clean
df_merged.drop(columns=['match_title', 'fixed_title'], inplace=True)

# 5. Save the final results!
output_file = "soc_code_match_results.csv"
df_merged.to_csv(output_file, index=False)

print(f"Success! Saved merged data to {output_file}")

# Validation Check
unmatched = df_merged[df_merged['O*NET-SOC Code'].isna() & df_merged['FINAL TITLE'].notna()]
print(f"Total rows matched: {len(df_merged) - len(unmatched)}")
print(f"Rows remaining unmatched (mostly due to typos or blank entries): {len(unmatched)}")
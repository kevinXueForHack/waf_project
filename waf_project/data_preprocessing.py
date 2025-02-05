import pandas as pd  # Data processing, CSV file I/O


# Set pandas display options for better visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)

# Load raw loan data
raw_ld_df = pd.read_csv("data/raw_loan_data_date.csv", low_memory=False)
unprocessed_ld_df = raw_ld_df.copy()

def adjust_year(dt):
    return dt.replace(year=dt.year - 100) if dt.year >= 2025 else dt

# Convert money signs to allow for future calculations
unprocessed_ld_df[['DISBURSEMENT_AMOUNT', 'CHARGE_OFF_AMOUNT', 'LOAN_AMOUNT', 'SBA_APPROVED_AMOUNT']] = unprocessed_ld_df[[
    'DISBURSEMENT_AMOUNT', 'CHARGE_OFF_AMOUNT', 'LOAN_AMOUNT', 'SBA_APPROVED_AMOUNT']].replace('[\\$,]', '', regex=True).astype(float)
date_format = "%d-%b-%y"
date_cols = ["APPROVAL_DATE", "DISBURSEMENT_DATE", "DEFAULT_DATE"]
for col in date_cols:
    unprocessed_ld_df[col] = unprocessed_ld_df[col].astype("string")
    unprocessed_ld_df[col] = pd.to_datetime(unprocessed_ld_df[col], format=date_format)
    unprocessed_ld_df[col] = unprocessed_ld_df[col].apply(adjust_year)

# Convert categorical columns from Y/N to Boolean
unprocessed_ld_df['IS_REVOLVER'] = unprocessed_ld_df['IS_REVOLVER'].map({'Y': True, 'N': False})
unprocessed_ld_df['IS_LOW_DOC'] = unprocessed_ld_df['IS_LOW_DOC'].map({'Y': True, 'N': False})

# Convert categorical columns into Boolean values
unprocessed_ld_df['IS_NEW'] = unprocessed_ld_df['IS_NEW'].map({1.0: True, 2.0: False})
unprocessed_ld_df['IS_URBAN'] = unprocessed_ld_df['IS_URBAN'].map({1: True, 2: False})
unprocessed_ld_df['LOAN_STATUS'] = unprocessed_ld_df['LOAN_STATUS'].map({"CHGOFF": False, "P I F": True})

# Create a copy of processed data before applying further filtering
processed_noReduced_ld_df = unprocessed_ld_df.copy()

# Drop rows where all economic indicators are missing
economic_indicators = [
    'TREASURY_YIELD', 'CPI_INDEX', 'GDP', 'MORTGAGE_30_US_FIXED',
    'UNRATE', 'INDPRO_INDEX', 'UMCSENT_INDEX', 'CSUSHPINSA_INDEX',
    'CP_INDEX', 'FEDFUNDS_RATE'
]
processed_noReduced_ld_df = processed_noReduced_ld_df.dropna(subset=economic_indicators, how='all')

# Drop rows with missing essential columns
processed_noReduced_ld_df = processed_noReduced_ld_df.dropna(subset=[
    'INDUSTRY_ID', 'IS_NEW', 'IS_URBAN', 'LOAN_STATUS', 'IS_REVOLVER',
    'BANK', 'BORROWER_NAME', 'IS_LOW_DOC', 'DISBURSEMENT_DATE'
])

# Remove rows where 'DISBURSEMENT_AMOUNT' is 0
processed_noReduced_ld_df = processed_noReduced_ld_df[processed_noReduced_ld_df['DISBURSEMENT_AMOUNT'] != 0]
processed_noReduced_ld_df['APPROVAL_DATE'] = pd.to_datetime(processed_noReduced_ld_df['APPROVAL_DATE'], errors='coerce')
processed_noReduced_ld_df = processed_noReduced_ld_df[processed_noReduced_ld_df['APPROVAL_DATE'].dt.year >= 1990]
processed_noReduced_ld_df = processed_noReduced_ld_df[processed_noReduced_ld_df['INDUSTRY_ID'] != 0]

# Create a copy for further row reduction processing
processed_row_reduced_ld_df = processed_noReduced_ld_df.copy()
processed_row_reduced_ld_df = processed_row_reduced_ld_df.drop(columns=['CITY', 'BALANCE_AMOUNT'])
processed_row_reduced_ld_df = processed_row_reduced_ld_df.reset_index(drop=True)
processed_row_reduced_ld_df['EXPOSURE'] = processed_row_reduced_ld_df['DISBURSEMENT_AMOUNT'] -  processed_row_reduced_ld_df['SBA_APPROVED_AMOUNT']
processed_row_reduced_ld_df['PERCENTAGE_EXPOSURE'] = processed_row_reduced_ld_df['EXPOSURE']/processed_row_reduced_ld_df ['DISBURSEMENT_AMOUNT']

# Display the final processed DataFrame information
processed_ld_df = processed_row_reduced_ld_df.copy()
processed_ld_df.to_csv('data/processed_loan_data.csv', index=False) 
I contributed 100% of the code in this project I submitted for a technical. When it comes to ChatGPT, I used it for mainly helping me understand ML/Pandas documentation and cleaning up my code and adding docstrings/useful comments to my messy final code. 


My actual cleaned up code is in the WAF_Project folder, while my notebooks are in the notebooks folder and are more experimental then actual code

THIS README IS JUST MY THOUGHT AND DOES NOT HAVE ANY USE

data preprocessing steps that I want to work on 
turning all the data into a specific format

5. is_revolver and is_low_doc seems to be Y and N for boolean
1. is revolver has a bunch of random T's everywhere and is empty
2. is_low_doc iseems ot have missing parts that are just 0's LOL
3. is_new seems to be a boolean factor using 1 and 2 instead normal boolean values
    2 appears significantly less than 1
    Assume 1 to be new loans, but treat this datapoint cautiously
1. convert is_urban and is_new to 1 true, 2 false, 0 old data
4. convert paid off to be a boolean yes for PIF and no for CHGOFF


row dimensionality reduction
2. deleteing all the empty treasury/cpi/gdp/MORTGAGE_30_US_FIXED	UNRATE	INDPRO_INDEX	UMCSENT_INDEX	CSUSHPINSA_INDEX	CP_INDEX	FEDFUNDS_RATE info
3. deleting all the empty industry_id info
6. deleting all Nan is_new, is_urban
7. deleting all $0 disimbursemnt amount
5. deleting all data before 1990


column dimensionality reduction
1. the bank state can lowkey go 
2. jobs created can go
3. jobs retained can go
4. balance can go
5. cities gotta go NGL just use zip code analysis instead

split the data into two sections:
1. paid off loans
2. non paid off loans

building multi-regression
building logistic regrssion
builidng k-clustering


3. cluster applications based on a combo of loan_size, market_conditions, and tern/exposre

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










5. there exists a lot of tables with no industry code near the early dates



6. there's a couple missing dates as well



split the data into two sections:
1. paid off loans
2. non paid off loans

EDA: Does the data have any interesting quirks or features? Describe any important
preprocessing/wrangling steps you took to clean the data.


do a linear regression for charge_off_amount


build 

Regression: What are the most predictive factors in determining the amount charged off? Brainstorm or
build a predictive regression model for the `CHARGE_OFF_AMOUNT` column. Explain how you
evaluate the performance of the model.

We want to multivaraite regression because it allows us to judge off more than one variabel
However we struggle with multicollinearity.
Thus, OLS sucks with multicollinearity, and PCR is bad at intrepretability
A good enough compromise is


Independent variables are presumably

1. Term
2. Employee Count
3. is_urban
4. is_low_doc
5. disbursement amount
6. percentage exposure
7. UMCSENT_INDEX
8. INDPRO_INDEX
9. FEDFUNDS_RATE

Classification: What are the most predictive factors in determining if a loan will default? On the flip side,
what are the most predictive factors in determining if a loan will be paid back in full? Brainstorm or build
a predictive classification model for the dataset.

do a logistic regression/classification model for paid-off/loan status dataset


Clustering: If you had to decide on 3-5 “types” or “groups” of loan applications, what would they be?
Describe your thought process in picking these groups, as well as models you build to implement this. If
you choose not to write a clustering model, please be prepared to defend your assumptions using other
data analysis techniques. (Note: this question is intentionally ambiguous – we encourage you to come up
with your own interpretation!)

So What?: What are your final recommendations for a lender trying to determine if a loan should be
approved or not? What are the insights and implications? Incorporate the analysis from the models used in
previous questions to recommend steps or action items to better predict loan defaults beyond using the
model itself. (Note: Once again, this question is open-ended and does not have one right answer. We’re
looking for creativity in how you might recommend the company to proceed!)



vibes:
1. build an OLS model for charge off amount
2. build a logistic/classification model for defaulting/PIF
3. cluster applications based on a combo of loan_size, market_conditions, and tern/exposre
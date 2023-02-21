# 
Loan-Default-Prediction-model by Manish Taori
We are developing a machine learning model that can predict the likelihood of a person defaulting on their loan within the next two years. The goal of this project is to provide assistance to lenders in making informed decisions regarding loan approvals based on the model's output.

## Project Intro/Objective
The main objective of this project is to provide assistance to lenders who are struggling with loan defaults by developing a predictive model that can aid them in making better-informed decisions.

### Methods Used
* Dataset Preprocessing (SQL)
* Simple Exploratory Data Analysis
* Feature Selection using Random Forrest and SelectKBest (f_classif)
* Predictive Modeling
* Evaluation Metrix

### Technologies
* Python
* MySql
* Pandas
* numpy 
* matplotlib
* seaborn
* Jupyter notebook

## Project Description
Financial institutions such as banks and multi-finance companies rely heavily on their lending activities to generate income. However, such activities pose a significant risk to lenders as borrowers may default on their loans, resulting in substantial financial losses. To minimize the risk of default, lenders need to carefully select qualified borrowers, determine appropriate interest rates, and decide on suitable loan amounts. This project aims to aid struggling lenders by developing a model that can assist them in making informed decisions.

## Important Concepts
Credit card and home loans are examples of lending services where a borrower receives funds from a lender. When we use a credit card, the money we spend is not our own, and we are required to pay it back to the lender, along with interest if we are unable to repay the debt in full. Home loans are another form of lending, where a borrower puts up collateral, such as their home, to guarantee the loan. Another example of credit is asset financing, where organizations can finance assets and pay for them over time rather than purchasing them outright.

### Column Description
Table | Columns | Description
----- | ------- | -----------
loanapptrain.csv / loanapptest.csv | LN_ID | Loan ID
loanapptrain.csv / loanapptest.csv | TARGET | Target variable ( 1 = client with late payment more than x days; 0 = all other cases)
loanapptrain.csv / loanapptest.csv | CONTRACT_TYPE | Identification if loan is cash or revolving
loanapptrain.csv / loanapptest.csv | GENDER | Gender of the client
loanapptrain.csv / loanapptest.csv | NUM_CHILDREN | Number of children the client has
loanapptrain.csv / loanapptest.csv | INCOME | Monthly income of the client
loanapptrain.csv / loanapptest.csv | APPROVED_CREDIT | Approved credit amount of the loan
loanapptrain.csv / loanapptest.csv | ANNUITY | Loan annuity (amount that must be paid monthly)
loanapptrain.csv / loanapptest.csv | PRICE | For consumer loans it is the price of the goods for which the loan is given
loanapptrain.csv / loanapptest.csv | INCOME_TYPE | Clients income type (businessman, working, maternity leave,...)
loanapptrain.csv / loanapptest.csv | FAMILY_STATUS | Family status of the client
loanapptrain.csv / loanapptest.csv | HOUSING_TYPE | What is the housing situation of the client (renting, living with parents,...)
loanapptrain.csv / loanapptest.csv | DAYS_AGE | Client's age in days at the time of application
loanapptrain.csv / loanapptest.csv | DAYS_WORK | How many days before the application the person started current job
loanapptrain.csv / loanapptest.csv | DAYS_REGISTRATION | How many days before the application did client change his registration
loanapptrain.csv / loanapptest.csv | DAYS_ID_CHANGE | How many days before the application did client change the identity document with which he applied for the loan
loanapptrain.csv / loanapptest.csv | WEEEKDAYS_APPLY | On which day of the week did the client apply for the loan
loanapptrain.csv / loanapptest.csv | HOUR_APPLY | Approximately at what hour did the client apply for the loan
loanapptrain.csv / loanapptest.csv | ORGANIZATION_TYPE | Type of organization where the client works
loanapptrain.csv / loanapptest.csv | EXT_SCORE_1 | Normalized score from the external data source
loanapptrain.csv / loanapptest.csv | EXT_SCORE_2 | Normalized score from external data source
loanapptrain.csv / loanapptest.csv | EXT_SCORE_3 | Normalized score from external data source
prevloanapp.csv | LN_ID_PREV | ID of previous loan (One loan can have 0,1,2 or more previous loan application)
prevloanapp.csv | LN_ID | Loan_ID
prevloanapp.csv | CONTRACT_TYPE | Contract product type (Cash loan, consumer loan,...) of the previous application
prevloanapp.csv | ANNUITY | Loan annuity (amount that must be paid monthly) of the previous application
prevloanapp.csv | APPLICATION | For how much credit did client ask on the previous application
prevloanapp.csv | APPROVED_CREDIT | Final approved credit ammount on the previous application. This differs from APPLICATION in a way that the APPLICATION is the ammount for which the client initially applied for, but during our approval process, he could have received differend amount (AMT_CREDIT)
prevloanapp.csv | AMT_DOWN_PAYMENT | Down payment on the previous application
prevloanapp.csv | PRICE | For consumer loans, it is the price of the goods for which the loan is given
prevloanapp.csv | WEEKDAYS_APPLY | On which day of the week did the client apply for the previous loan
prevloanapp.csv | HOUR_APPLY | Approximately at what hour did the client apply for the previous loan
prevloanapp.csv | CONTRACT_STATUS | Contract status (approved, cancelled,...) of previous application
prevloanapp.csv | DAYS_DECISION | Relative to current application when was the decision about previous application made.
prevloanapp.csv | TERM_PAYMENT | Term of previous credit at application of the previous application
prevloanapp.csv | YIELD_GROUP | Grouped interest rate into small, medium and high of the previous application
prevloanapp.csv | FIRST_DRAW | Relative to application date of current application when was the first disbursement of the previous application (in days)
prevloanapp.csv | FIRST_DUE | Relative to application date of current application when was the first due supposed to be of the previous application (in days)
prevloanapp.csv | TERMINATION | Relative to application date of current application when was the expected termination of the previous application
prevloanapp.csv | NFLAG_INSURED_ON_APPROVAL | Did the client requested insurance during the previous application
installment_payment.csv | LN_ID_PREV | ID of previous loan (One loan can have 0,1,2 or more previous loan application)
installment_payment.csv | LN_ID | Loan ID
installment_payment.csv | INST_NUMBER | On which installment we observe payment
installment_payment.csv | INST_DAYS | When the installment of previous credit was supposed to be paid (relative to application date of current loan)
installment_payment.csv | PAY_DAYS | When was the installments of previous credit paid actually (relative to application date of current loan)
installment_payment.csv | AMT_INST | What was the prescribed installment amount of previous credit on this installment
installment_payment.csv | AMT_PAY | What the client actually paid on previous credit on this installment

## Getting Started

1. Dataset Preparation
    - We've tried to import raw dataset directly using Python Library `(Pandas and Dask)` but we encountered problems due to our less sufficiency memory size. We decided to use another method.
    - We decided to do formatting the raw CSV dataset using both Microsoft Excel and SQL combined.
    - First, we did CSV formatting using Microsoft Excel, replaced the blank values with Null to avoid truncated data warning in SQL, removed thousand separator and then saved it.
    - After that, we imported the formatted CSV to SQL using `LOAD DATA INFILE` Query. The query was succesfull. In the end, we got 6 tables in 1 schema as equal to 6 raw CSV data we received.
    - After we joined some tables, we exported them into new sql and csv data. Then we proceed to Exploratory Data Analysis step. We decided to limit data rows for 15000 rows due to efficiency reason
2. Exploratory Data Analysis
    - Importing new formatted CSV
    - Descriptive Analysis
    - Client/Customer's Profiling
    - Client/Customer's Behaviour
3. Preprocessing
    - Value Encoding: This step is for preparing the dataset to be ready feature selected and modelled
    - Feature Selection : Correlation Analysis, Random Forrest, and SelectKBest (f_classif)
    - From the feature selection, we understand that; 'EXT_SCORE_1 is the most important feature in this credit risk modelling as followed by 'EXT_SCORE_3' and 'EXT_SCORE_2'
4. Model Building
    - Handling Imbalance Target
        - Oversampling using SMOTE
        - Undersampling using NearMiss
        - We knew that Oversampling technique has far better result than undersampling in handling imbalanced dataset. So, we decided to use oversampling technique in advance.
    - Logistic Regression before and after Tuning
    - Random Forrest Classifier before and after Tuning
    - Decision Tree Classifier before and after Tuning

### Evaluation Metrics in Data Train
#### 1. Random Forrest Before Tuning
No | Metrics | Score
-- | ------- | -----
1 | Accuracy | 0.84
2 | Recall | 0.16
3 | Precision | 0.12
4 | ROC AUC Score | 0.53
5 | F1 Score | 0.14
#### 2. Neural Network
No | Metrics | Score
-- | ------- | -----
1 | Accuracy | 0.92
2 | Recall | 0
3 | Precision | 0
4 | ROC AUC Score | 0.50
5 | F1 Score | 0

### Evaluation Metrics in Data Test
#### 1. Random Forrest
No | Metrics | Score
-- | ------- | -----
1 | Accuracy | 0.80
#### 2. Neural Network
No | Metrics | Score
-- | ------- | -----
1 | Accuracy | 0.55

## Conclusion
1. Neural Network with Oversampling Technique Algorithm (SMOTE) has better accuracy (92%) for the train dataset than Logistic Regression, Random Forrest and Decision Tree in both after tuned or before tuned. However, Evaluation Metrics show that Random Forrest algorithm has better accuracy, which is 80.24% in Test Dataset than Neural Network. 

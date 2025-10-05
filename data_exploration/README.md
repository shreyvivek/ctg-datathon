# Data Exploration & Cleanup (summary)
- Fixed headers from CTG.xls (row 2 as header), dropped empty cols.
- Removed leakage (`CLASS`).
- Kept required 23 features for grading: {b,e,AC,FM,UC,DL,DS,DP,DR,LB,ASTV,MSTV,ALTV,MLTV,Width,Min,Max,Nmax,Nzeros,Mode,Mean,Median,Variance,Tendency}.
- Imputed numeric missing values with column means; noted class imbalance.

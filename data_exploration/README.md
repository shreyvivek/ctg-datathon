# Data Exploration & Cleanup (summary)
- Loaded CTG Excel, fixed headers (row 2 as header), dropped empty cols.
- Removed leakage (`CLASS`) from features.
- Kept required 23 features for grading:
  {b,e,AC,FM,UC,DL,DS,DP,DR,LB,ASTV,MSTV,ALTV,MLTV,Width,Min,Max,Nmax,Nzeros,Mode,Mean,Median,Variance,Tendency}
- Imputed numeric missing values with column means.
- Checked class imbalance (Suspect/Pathologic are minority).

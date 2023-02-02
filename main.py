import pandas as pd
import evaluation, classifiers, pca, classifiers_with_pca

#read csv to dataframe
df = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
#change specified data types to category
for col in ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']:
    df[col] = df[col].astype('category')

#Question 1: Data Evaluation
evaluation.prints(df)

#Question 2: Use Classifiers and Compare to Paper
#create new dataframe excluding the time feature
df_new = df.drop('time', axis=1)
classifiers.knn(df_new)
classifiers.random_forest(df_new)

#Question 3 & 4: PCA & Cumulative Explained Variance
pca.rankFeatures(df_new)

#Question 5: Classifiers using PCA Components
classifiers_with_pca.rf_2pc(df_new)
classifiers_with_pca.rf_10pc(df_new)
classifiers_with_pca.gb_2pc(df_new)
classifiers_with_pca.gb_10pc(df_new)
classifiers_with_pca.svmr_2pc(df_new)
classifiers_with_pca.svmr_10pc(df_new)

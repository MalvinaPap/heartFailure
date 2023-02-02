def prints(df):
    #printings for numeric values
    print('---------------------------------------------------')
    print("\nNUMERIC VALUES EVALUATION\n")
    
    print("\nCount of samples according to survival:")
    print(df['DEATH_EVENT'].value_counts(dropna=False))

    print("\nStats for numeric values (whole dataset):")
    print(df.describe(exclude="category").transpose())
    
    #separate dataframe according to survival of patients
    df_survived= df.loc[df['DEATH_EVENT'] == 0]
    df_died= df.loc[df['DEATH_EVENT'] == 1]

    print("\nStats for numeric values (survived patients):")
    print(df_survived.describe(exclude="category").transpose())

    print("\nStats for numeric values (dead patients):")
    print(df_died.describe(exclude="category").transpose())

    #printings for category values
    print('---------------------------------------------------')
    print("\nCATEGORY VALUES EVALUATION\n")

    print("\nCount of samples according to Anaemia:")
    print(df['anaemia'].value_counts(dropna=False))

    print("\nCount of samples according to Diabetes:")
    print(df['diabetes'].value_counts(dropna=False))

    print("\nCount of samples according to High Blood Pressure:")
    print(df['high_blood_pressure'].value_counts(dropna=False))

    print("\nCount of samples according to Sex:")
    print(df['sex'].value_counts(dropna=False))

    print("\nCount of samples according to Smoking")
    print(df['smoking'].value_counts(dropna=False))
    print('---------------------------------------------------')
 

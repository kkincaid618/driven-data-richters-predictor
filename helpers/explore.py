from pandas import DataFrame

class ExploreData():
    def evaluate_class_balance(y):
        counts = DataFrame(y.value_counts(),columns=['counts'])
        counts['perc'] = round(counts['counts'] / len(y) * 100, 1)
        
        return counts

    def evaluate_relationships(df,cols):
        sm = df.melt(id_vars='damage_grade',value_vars=cols)

        med = sm.groupby(['variable','damage_grade']).median()
        mean = sm.groupby(['variable','damage_grade']).mean()
        std = sm.groupby(['variable','damage_grade']).std()

        all = med.merge(mean,left_index=True,right_index=True)
        all = all.merge(std,left_index=True,right_index=True)

        all.columns = ['median','mean','std']

        return all

    def evaluate_nulls(df):
        df_null = DataFrame(df.isna().sum())
        df_null.columns = ['nulls']
        df_null = df_null[df_null['nulls'] > 0]
        df_null['perc'] = round(df_null['nulls'] / len(df) * 100,2)

        return df_null
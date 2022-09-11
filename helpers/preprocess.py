from pandas import DataFrame, qcut, cut, get_dummies
from numpy import log, where, NaN, floor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from datetime import date

class DataProcessor():
    def __init__(self, X, y, val_data, resample_flag=False, subsequent_run_flag=False):
        self.X = X
        self.y = y
        self.val_data = val_data
        self.resample_flag = resample_flag
        self.subsequent_run_flag = subsequent_run_flag

    def convert_dtypes(self):
        X = self.X
        val_data = self.val_data

        binary_cols = [col for col in X.columns if 'has_' in col]
        int_cols = [col for col in X.columns if 'count_' in col or '_percentage' in col or 'age' == col]
        cat_cols = [col for col in X.columns if col not in binary_cols and col not in int_cols]

        X[int_cols] =  X[int_cols].astype('int')
        X[cat_cols] = X[cat_cols].astype('str')
        X[binary_cols] = X[binary_cols].astype('str')

        val_data[int_cols] =  val_data[int_cols].astype('int')
        val_data[cat_cols] = val_data[cat_cols].astype('str')
        val_data[binary_cols] = val_data[binary_cols].astype('str')

        print('[INFO] Columns have been transformed into their appropriate data types')

        return X, val_data

    def transform_data(self, cols, type):
        X = self.X
        val_data = self.val_data

        X_transform = X.loc[:,cols]
        val_transform = val_data.loc[:,cols]

        if type=='log':
            X_transform = log(X_transform + 1)
            val_transform = log(val_transform + 1)
        elif type=='reciprocal':
            X_transform = 1 / (X_transform + 1)
            val_transform = 1 / (val_transform + 1)

        X = X.drop(cols,axis=1)
        val_data = val_data.drop(cols,axis=1)

        X = X.merge(X_transform,left_index=True,right_index=True)
        val_data = val_data.merge(val_transform,left_index=True,right_index=True)

        return X, val_data

    def replace_outliers(self, cols, value_to_replace, replacement_value):
        X = self.X
        val_data = self.val_data

        X[cols] = where(X[cols] == value_to_replace, replacement_value, X[cols])
        val_data[cols] = where(val_data[cols] == value_to_replace, replacement_value, val_data[cols])

        print(f'[INFO] Value {value_to_replace} has been replaced with {replacement_value} in {cols}')

        return X, val_data

    def drop_cols(self,cols):
        X = self.X
        val_data = self.val_data

        X = X.drop(cols,axis=1)
        val_data = val_data.drop(cols, axis=1)

        print(f'[INFO] Columns {cols} have been dropped from the data frame')

        return X, val_data

    def scale(self, scaler, num_cols):
        scaler = scaler

        X_train = self.X_train
        X_test = self.X_test
        val_data = self.val_data

        X_train_num = X_train.loc[:,num_cols]
        X_test_num = X_test.loc[:,num_cols]
        val_data_num = val_data.loc[:,num_cols]

        X_train_sc = scaler.fit_transform(X_train_num)
        X_train_sc = DataFrame(X_train_sc,columns=X_train_num.columns,index=X_train_num.index)

        X_test_sc = scaler.transform(X_test_num)
        X_test_sc = DataFrame(X_test_sc,columns=X_test_num.columns,index=X_test_num.index)

        val_data_sc = scaler.transform(val_data_num)
        val_data_sc = DataFrame(val_data_sc,columns=val_data_num.columns,index=val_data_num.index)

        X_train_str = X_train.drop(num_cols,axis=1)
        X_test_str = X_test.drop(num_cols,axis=1)
        val_data_str = val_data.drop(num_cols,axis=1)

        X_train = X_train_sc.merge(X_train_str,left_index=True,right_index=True)
        X_test = X_test_sc.merge(X_test_str,left_index=True,right_index=True)
        val_data = val_data_sc.merge(val_data_str,left_index=True,right_index=True)

        print(f'[INFO] The numeric features {num_cols} were scaled')

        return X_train, X_test, val_data

    def split(self, split_size):
        X = self.X
        y = self.y

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=split_size, stratify=y)

        print(f'[INFO] Data has been split into training & test sets')
        print(f'[INFO] X_train = {X_train.shape}')
        print(f'[INFO] y_train = {y_train.shape}')
        print(f'[INFO] X_test = {X_test.shape}')
        print(f'[INFO] y_test = {y_test.shape}')

        return X_train, X_test, y_train, y_test

    def bin(self,bin_cols,num_q):
        X_train = self.X_train
        X_test = self.X_test
        val_data = self.val_data

        label_values = []

        for i in range(num_q):
            label = 'bin_' + str(i+1)
            label_values.append(label)

        X_train[bin_cols], bins = qcut(X_train[bin_cols], q=num_q, retbins=True, labels=label_values)
        X_test[bin_cols] = cut(X_test[bin_cols], bins=bins, labels=label_values)
        val_data[bin_cols] = cut(val_data[bin_cols], bins=bins, labels=label_values)

        print(f'[INFO] {num_q} bins were created for {bin_cols} columns and applied to test set')

        return X_train, X_test, val_data
    
    def dummy_cols(self,dum_cols):
        X_train = self.X_train
        X_test = self.X_test
        val_data = self.val_data

        X_train_dum = get_dummies(X_train.loc[:,dum_cols],dummy_na = False)
        X_test_dum = get_dummies(X_test.loc[:,dum_cols],dummy_na = False)
        val_data_dum = get_dummies(val_data.loc[:,dum_cols],dummy_na = False)

        X_train.drop(dum_cols,axis=1,inplace=True)
        X_test.drop(dum_cols,axis=1,inplace=True)
        val_data.drop(dum_cols,axis=1,inplace=True)

        X_train = X_train.merge(X_train_dum,left_index=True,right_index=True)
        X_test = X_test.merge(X_test_dum,left_index=True,right_index=True)
        val_data = val_data.merge(val_data_dum,left_index=True,right_index=True)

        X_train, X_test = X_train.align(X_test, join="outer", axis=1,fill_value=0)
        X_train, val_data = X_train.align(val_data, join="left", axis=1, fill_value=0)

        print(f'[INFO] {len(dum_cols)} columns have been dummied: {dum_cols}')
        print(f'[INFO] X_train = {X_train.shape}')
        print(f'[INFO] X_test = {X_test.shape}')
        print(f'[INFO] val_data = {val_data.shape}')

        return X_train, X_test, val_data

    def resample(self, algorithm):
        X_train = self.X_train
        y_train = self.y_train

        sampler = algorithm

        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

        print('[INFO] Target variable has been resampled')
        print(f'[INFO] X_train = {X_train_resampled.shape}')
        print(f'[INFO] y_train = {y_train_resampled.shape}')

        return X_train_resampled, y_train_resampled

    def drop_nonuniques(self):
        X_train = self.X_train
        X_test = self.X_test
        val_data = self.val_data

        drop_cols = [col for col in X_train.columns if X_train[col].nunique() == 1]
        X_train = X_train.drop(drop_cols, axis=1)
        X_test = X_test.drop(drop_cols, axis=1)
        val_data = val_data.drop(drop_cols, axis=1)

        print(f'[INFO] {len(drop_cols)} columns were dropped for being non-unique: {drop_cols}')

        return X_train, X_test, val_data

    def group_values(self, column, infrequency_threshold_perc):
        X_train = self.X_train
        X_test = self.X_test
        val_data = self.val_data

        n_records_cutoff = floor(infrequency_threshold_perc*len(X_train))
        values = DataFrame(X_train[column].value_counts())
        infrequents = values[values[column] <= n_records_cutoff].index.tolist()

        all_values = X_train[column]
        X_train[column] = where(X_train[column].isin(infrequents), 'rare', X_train[column])
        X_test[column] = where((X_test[column].isin(infrequents)) | ~(X_test[column].isin(all_values)),
                            'rare',X_test[column])
        val_data[column] = where((val_data[column].isin(infrequents)) | ~(val_data[column].isin(all_values)),
                            'rare',val_data[column])

        return X_train, X_test, val_data

    def process(self):
        binary_cols = [col for col in self.X.columns if 'has_' in col]
        drop_cols = []
        int_cols = [col for col in self.X.columns if 'count_' in col or '_percentage' in col]
        dum_cols = [col for col in self.X.columns if col not in (int_cols + drop_cols + binary_cols)]

        if self.subsequent_run_flag == False:
            self.X, self.val_data = self.convert_dtypes()
        
        self.X, self.val_data = self.replace_outliers(['age'],995,NaN)
        self.X, self.val_data = self.drop_cols(drop_cols)
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.split(0.70)
        self.X_train, self.X_test, self.val_data = self.bin('age',3)
        self.X_train, self.X_test, self.val_data = self.group_values('geo_level_3_id',0.0001)
        self.X_train, self.X_test, self.val_data = self.dummy_cols(dum_cols)
        self.X_train, self.X_test, self.val_data = self.drop_nonuniques()
    
        if self.resample_flag == True:
            self.X_train, self.y_train = self.resample(SMOTE())

        self.X_train, self.X_test, self.val_data = self.scale(StandardScaler(), int_cols)

        print('[INFO] Data has been processed according to configuration variables')

        return self.X_train, self.X_test, self.y_train, self.y_test, self.val_data

class PredictionGenerator():
    def __init__(self, model, model_type, X_val):
        self.model = model
        self.X_val = X_val
        self.model_type = model_type

    def predict(self):
        model = self.model
        X_val = self.X_val

        y_pred = model.predict(X_val)
        
        return y_pred
    
    def generate_path(self):
        model_type = self.model_type
        today = date.today().strftime("%Y%m%d")

        path = f'./predictions/kk-predictions-{model_type}-{today}.csv'

        return path

    def save_predictions(self):
        self.y_pred = self.predict()
        self.path = self.generate_path()

        submission = DataFrame(self.y_pred,index=self.X_val.index,columns=['damage_grade'])
        submission.to_csv(self.path)

        print(f'Predictions saved to {self.path}')
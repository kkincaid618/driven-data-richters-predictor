from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import f1_score, roc_auc_score

class TrainModels():
    def __init__(self, models, X_train, X_test, y_train, y_test):
        self.models_to_train = models
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluate_models(self,model_name, y_true, y_pred):
        score = f1_score(y_true, y_pred, average='micro')
        print(f'F1 Score for {model_name} is {score}')
    
    def train_models(self):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train.values.ravel()
        y_test = self.y_test

        models_to_train = self.models_to_train

        for model_name in models_to_train:
            model = models_to_train[model_name]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            self.evaluate_models(model_name,y_test,y_pred)

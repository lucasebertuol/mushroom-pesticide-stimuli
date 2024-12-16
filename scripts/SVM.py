from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold, cross_val_predict, LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, confusion_matrix
from sklearn.svm import SVC, LinearSVC

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#--------------------------------------------------------
class SVM:

    #--------------------------------------------------------
    def __init__(self):
        pass     
      
    #--------------------------------------------------------
    def create_svm_model(self, x, y):
    
        param_grid = {
                'C': [0.001, 0.01, 0.1, 1],   
                'max_iter': [10000],       
                'penalty': ['l2'],                    
                'class_weight': ['balanced'],
                'dual': [True],                     
        }     

        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        model = LinearSVC()
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=stratified_kfold, n_jobs=-1, scoring='accuracy')
        grid_search.fit(x, y)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print("Best parameters:", best_params)

        model = grid_search.best_estimator_

        y_pred = cross_val_predict(model, x, y, cv=stratified_kfold)
        
        cm = confusion_matrix(y, y_pred)

        tn, fp, fn, tp = cm.ravel()
        
        annot = np.array([[f'TP: {tp}', f'FN: {fn}'], [f'FP: {fp}', f'TN: {tn}']])
        
        print(annot)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, 
                    xticklabels=[r'$\bf{Herbicide}$', r'$\bf{Water}$'], 
                    yticklabels=[r'$\bf{Herbicide}$', r'$\bf{Water}$'],
                    annot_kws={'size': 16})
        plt.ylabel('Ground truth', fontweight='bold',  fontsize=16)
        plt.xlabel('Predictions', fontweight='bold',  fontsize=16)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)


        plt.savefig('svm.png', dpi=300, bbox_inches='tight')
        
        print('TN: %d FP: %d FN: %d TP: %d' % (tn, fp, fn, tp))
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) 
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        print("Accuracy: ", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("Specificity:", specificity)
        print("F1 Score: ", f1)
  
        
    #--------------------------------------------------------

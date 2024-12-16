from LoadData import LoadData
from SVM import SVM

import Constants as C
#-----------------------------------------------------------------------------
def main():

    ld = LoadData() 
    x = ld.load_signals(C.DATA)
    y = ld.load_classes(C.CLASSES)
    
    x_scaled = ld.scaler(x)
    
    print('----------------------------------------')
    print('---------------- SVM -------------------')
    print('----------------------------------------')
    SVM().create_svm_model(x, y) 
  
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()

from sklearn import svm

lin_clf = svm.LinearSVC()
lin_clf.fit(Xtrain, Ytrain) 
svm_result = lin_clf.predict(Xtest)
np.savetxt("svm_result.csv", np.dstack((np.arange(1, svm_result.size+1),svm_result))[0],"%d,%d",header="Id,ClassLabel")

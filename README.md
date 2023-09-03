# Credit-Card-Fraud-Detection

This project was a part of a CodSoft internship in which I had to perform 3 tasks out of which the third was Credit Card Fraud Detection. In this project, I had to build a machine-learning model to identify fraudulent credit card transactions.







At the start, the dataset was imported with the help of Pandas Library, and then preliminary checking of the dataset was performed to check for any outliers. After that, the fraud and correct transactions were distinguished with the help of Class values and were plotted to see a clear view of fraud and correct transactions. The data set was further split using train_test_split for training and testing the machine learning model such that the transactions and class values were separated into two variables to train the machine learning model. Before training the model, the preprocessing and normalization are done using StandardScalar(), which scales each feature (column) in the dataset to have a mean of 0 and a standard deviation of 1. This means that after applying StandardScaler, each feature will have a similar scale, making it easier for machine learning algorithms to work with the data. Further, the Machine learning models and metrics for measuring their quality were imported according to the assigned task. After that, each model was built and predicted according to different metrics to see which would be more effective or good for prediction.



Further, the class imbalance issue was dealt with both oversampling and undersampling techniques and again the whole process was performed in order to check which conditions give the optimum results it can be seen that almost all the models oversampled dataset outputs are better than the undersampled dataset also.


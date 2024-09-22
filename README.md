"# Spam-SMS-classifier-NLTK-" 

This project is based on Supervised learning. Spam sms classifier model is trained to identify
spam sms from other sms by training them on features of such spam sms and then testing the
model to find the accuracy of its classification against various classifiers.
The data set comes from the UCI Machine Learning Repository. It contains over 5000 SMS
labelled messages, tagged accordingly, with being ham (legitimate) or spam, that have been
collected for mobile phone spam research. It is downloaded from the following URL:
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

"#Data Pre-processing"
Pre-processing the data is an essential step in natural language process.
1. Class labels are converted to binary values using the LabelEncoder from sklearn.
2. Further, email addresses, URLs, phone numbers, and other symbols are replaced using
regular expressions.
3. Stop words are removed and word stems are extracted.

"#Generating Features"
Feature engineering is the process of using domain knowledge of the data to create features for
machine learning algorithms. In this project, the words in each text message will be our
features. For this purpose, each word is tokenized and the 1500 most common words are used
as features.

"#Scikit-Learn Classifiers with NLTK"
Various classifiers are imported from sklearn. Some performance metrics, such as
accuracy_score and classification_report are also imported.

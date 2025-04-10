# Documentation for this Directory

## File
See **handlingImbalancing.py** to read relavant code

## Progress Report
The current process of creating a decision tree model is outlined below
- Handle Inbalancing FIRST using SMOTE (Synthetic Minority Over-sampling Technique)
- Splitting the provided data into a training and a testing set
    -   This creates new samples for less represented classes, so that there are an equal number of samples for each class (not exact duplicates)
- Do RFE (Recursive Feature Elimination); not sure what the exact differences are between this and backwards elimination; at the moment, the elimination rate is set to 20% - so every round 20% of remaining features are eliminated
- Do cross validation for the chosen features
    - Not sure if this is actually necessary because RFE should already be choosing the features based on the training data already
    - Might integrate CV into the RFE process later
- Train a model using the chosen features and the training data
- Use the test set to evaluate the model using metrics like f1-score, f1-score (weighted), f1-score (macro) and accuracy

## Doubts
- Should I split the data into training and testing sets before or after SMOTE - I tried to do it before but I got an error (because some classes were not present enough in the data for SMOTE to run)

- The evaluation looks very good (Accuracy, macro f1, weighted f1 ~ 0.88) and worst performance is on CLASS 24 with (precision, recall, f1-score) = (0.75, 0.74, 0.75). Can I actually trust these metrics to mean that the decision tree is a good fit for the data?
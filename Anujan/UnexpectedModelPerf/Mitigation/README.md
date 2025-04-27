# How to run the Mitigation Suite

NOTE: Ensure you are in the **./Mitigation** directory

1. First call the OriginalXGBoostAlgo_WithPSI.py with **python3 OriginalXGBoostAlgo_WithPSI.py** which will create a  model file called **parent_model.json**.

2. Next call TransferLearning_WithPSI.py with **python3 TransferLearning_WithPSI.py** which will create a model called  **learning_transfer.json**.

3. Finally, to generate predictions on the unlabelled test 2 dataset, execute FinalResults.py with **python3 FinalResults**, which will create the requested **preds_2.npy** file.
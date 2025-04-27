import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from crossWeightedEntropyLoss import weighted_log_loss

from sklearn.metrics import classification_report

from chosenFeatures import chosenFeatures


if __name__ == "__main__":
    # Loading in the original XGBoost model created from the original training data
    model_source = xgb.Booster(model_file='parent_model.json')

    # Add some duplicate samples so every class appears at least twice in the 
    # data. Saved this 'fudged' data in _edited.csv files.
    X_target = pd.read_csv("./data/X_test_2_edited.csv")
    y_target = pd.read_csv("./data/y_test_2_reduced_edited.csv")

    X_target = np.array(X_target)
    y_target = np.array(y_target)

    # Selecting the chosen features
    X_target = X_target[: , chosenFeatures]

    y_target = y_target.reshape(-1)

    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X_target, y_target, test_size=0.2, random_state=123, stratify=y_target
    )


    # Applying the same XGBoost techniques as the original model
    classes = np.unique(y_target_train)

    class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=y_target_train)

    class_weight_dict = dict(zip(classes, class_weights_array))
    sample_weights = np.array([class_weight_dict[label] for label in y_target_train])


    xgb_2 = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(classes),
        eval_metric='mlogloss',
        use_label_encoder=False,
        max_depth=3,
        reg_alpha=2,
        random_state=42,
        learning_rate=0.05, 
        n_estimators=500,
        min_child_weight=10,
    )

    # Fitting the new training data to the XGBoost model.
    # Notice that we provide the older XGBoost model to the fitting process
    # Improves learning performance
    xgb_2.fit(X_target_train, y_target_train, xgb_model=model_source, sample_weight=sample_weights)

    y_pred_proba = xgb_2.predict_proba(X_target_test)


    y_true = np.zeros((y_target_test.shape[0], 28))
    y_true[np.arange(y_target_test.shape[0]), y_target_test] = 1
    print(f"WEIGHTED LOG LOSS = {weighted_log_loss(y_true, y_pred_proba)}")


    y_pred = xgb_2.predict(X_target_test)
    print(y_pred)

    y_true=np.argmax(y_true, axis=1).reshape(-1)

    print(classification_report(y_true, y_pred, zero_division=0))

    xgb_2.save_model("learning_transfer.json")


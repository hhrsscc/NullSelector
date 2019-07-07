# NullSelector

Python implementation of feature selection with null importances.

This implementation's interfaces are interface scikit-learn like.


Examples:

rf = RandomForestClassifier(random_state=random_state)

selector = NullSeletor(rf)

selector.fit(train_x, train_y, features)

train_x = selector.transform(train_x)

valid_x = selector.transform(valid_x)


References:

https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
https://academic.oup.com/bioinformatics/article/26/10/1340/193348
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Splitting the dataset into two sets
# From data -> creating x_train and x_test
# From labels -> creating y_train and y_test
# Test size is the size of the test set (20% of the dataset)
# Shuffle the data before splitting -> avoid biases
# Stratify -> keep the same proportion of classes in train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

score = accuracy_score(y_pred, y_test)
print(f'{score * 100}% of samples were classified correctly')

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
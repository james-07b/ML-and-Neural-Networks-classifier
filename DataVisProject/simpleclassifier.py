from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd


def logistic_regression_thing(uri):
    df = pd.read_csv(uri)
    df.columns = df.columns.to_series().apply(lambda x: x.strip())

    my_df = df[['tweet', 'score']]
    my_df = my_df[my_df['score'] != 2]

    index = df.index
    df['random_number'] = pd.np.random.randn(len(index))
    train = df[df['random_number'] <= 0.8]
    test = df[df['random_number'] > 0.8]

    vector_machine = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = vector_machine.fit_transform(train['tweet'])
    test_matrix = vector_machine.transform(test['tweet'])

    lregress = LogisticRegression()
    X_train = train_matrix
    X_test = test_matrix
    y_train = train['score']
    y_test = test['score']

    lregress.fit(X_train, y_train)
    predictions = lregress.predict(X_test)

    new = pd.np.asarray(y_test)
    confusion_matrix(predictions, y_test)

    # saving accuracy value

    exact_log_acc = metrics.accuracy_score(y_test, predictions)
    log_reg_accuracy = "{:.2f}".format(exact_log_acc)
    print(log_reg_accuracy)

    print(classification_report(predictions, y_test))
    print("I am this much sure ",log_reg_accuracy)


logistic_regression_thing('data/steph.csv')






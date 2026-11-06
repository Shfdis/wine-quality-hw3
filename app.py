import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn import metrics


ROOT = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(ROOT / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


# Модель хранится как один общий объект, а cache_data возвращает копии таблиц.
# Без декоратора joblib.load читал бы файл при каждом движении ползунка.
model = load_model()
test = load_test()
features = list(model.feature_names_in_)
X_test = test[features]
y_test = test['target']
p_test = test_proba(model, X_test)
default_threshold = json.loads((ROOT / 'config.json').read_text())['threshold']

st.title('Подбор хороших вин')
threshold = st.sidebar.slider('Порог рекомендации', 0.0, 1.0, float(default_threshold), 0.01)
single, quality, batch, coefficients = st.tabs(
    ['Одно вино', 'Модельные метрики', 'Загрузка CSV', 'Коэффициенты'])

with single:
    values = {}
    for feature in features:
        if feature == 'is_red':
            values[feature] = int(st.toggle('Красное вино', value=bool(X_test[feature].median())))
        else:
            values[feature] = st.number_input(
                feature, min_value=float(X_test[feature].min()),
                max_value=float(X_test[feature].max()),
                value=float(X_test[feature].median()))
    probability = model.predict_proba(pd.DataFrame([values], columns=features))[0, 1]
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    st.write('Рекомендовать' if probability >= threshold else 'Не рекомендовать')

with quality:
    prediction = p_test >= threshold
    st.write('Матрица ошибок')
    fig, ax = plt.subplots()
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, prediction, ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    columns = st.columns(3)
    columns[0].metric('Precision', f'{metrics.precision_score(y_test, prediction, zero_division=0):.3f}')
    columns[1].metric('Recall', f'{metrics.recall_score(y_test, prediction, zero_division=0):.3f}')
    columns[2].metric('F1', f'{metrics.f1_score(y_test, prediction, zero_division=0):.3f}')

    fpr, tpr, _ = metrics.roc_curve(y_test, p_test)
    pr, rc, _ = metrics.precision_recall_curve(y_test, p_test)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr)
    axes[0].scatter(metrics.confusion_matrix(y_test, prediction)[0, 1] / sum(y_test == 0),
                    metrics.recall_score(y_test, prediction), color='red')
    axes[0].set(xlabel='FPR', ylabel='TPR', title='ROC')
    axes[1].plot(rc, pr)
    axes[1].scatter(metrics.recall_score(y_test, prediction),
                    metrics.precision_score(y_test, prediction, zero_division=0), color='red')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='PR')
    st.pyplot(fig)
    plt.close(fig)
    st.write('Если поднять порог, вин в подборке станет меньше. Если снизить, их станет больше, но посредственные будут попадать чаще.')

with batch:
    file = st.file_uploader('CSV с признаками вин', type='csv')
    if file is not None:
        uploaded = pd.read_csv(file)
        missing = [feature for feature in features if feature not in uploaded.columns]
        if missing:
            st.error('Не хватает столбцов: ' + ', '.join(missing))
        else:
            try:
                result = uploaded.copy()
                result['probability'] = model.predict_proba(uploaded[features])[:, 1]
                result['recommend'] = result['probability'] >= threshold
                st.dataframe(result)
                st.download_button('Скачать результат', result.to_csv(index=False), 'predictions.csv', 'text/csv')
            except ValueError as error:
                st.error(f'Проверьте значения признаков: {error}')

with coefficients:
    weights = pd.Series(model.named_steps['logisticregression'].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots()
    weights.plot.barh(ax=ax)
    ax.set_xlabel('Коэффициент после масштабирования')
    st.pyplot(fig)
    plt.close(fig)
    st.write(f'Сильнее всего за хорошее вино: {weights.index[-1]}. Против: {weights.index[0]}.')

st.caption('Данные: [Wine Quality, Cortez et al. (2009)](https://archive.ics.uci.edu/dataset/186/wine+quality), CC BY 4.0.')

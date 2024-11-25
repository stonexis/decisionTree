import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


def comparison_tree():
    # Загрузка данных
    df = pd.read_csv('TRAIN.csv')
    df = df.drop(df.columns[0], axis=1)

    # Числовые столбцы
    num_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
    cat_columns = ['cut', 'color', 'clarity']

    # Обработка числовых данных
    df[num_columns] = df[num_columns].apply(pd.to_numeric, errors="coerce")
    df[num_columns] = df[num_columns].fillna(0)  # Заменить NaN в числовых данных

    # Обработка категориальных данных
    df[cat_columns] = df[cat_columns].fillna("unknown")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    trainX_cat = encoder.fit_transform(df[cat_columns])

    # Создание массивов
    trainX_num = df[num_columns].to_numpy()
    trainY = df["price"].to_numpy()

    # Восстановление порядка столбцов
    trainX = np.hstack([trainX_cat, trainX_num])
    X_shuffled, y_shuffled = shuffle(trainX, trainY, random_state=42)

    # Гиперпараметры для тестирования
    hyperparams = [
        {"criterion": "squared_error", "max_depth": 12},
        {"criterion": "friedman_mse", "max_depth": 16},
        {"criterion": "poisson", "max_depth": 22},
        {"criterion": "squared_error", "max_depth": 45},
        {"criterion": "friedman_mse", "max_depth": 95},
        {"criterion": "poisson", "max_depth": 33},
    ]

    best_score = -np.inf
    best_params = None

    # Проверка каждой комбинации гиперпараметров
    for params in hyperparams:
        model = DecisionTreeRegressor(
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            random_state=42
        )

        # Кросс-валидация
        scores = cross_val_score(
            model, X_shuffled, y_shuffled, cv=10, scoring="r2"
        )

        mean_score = np.mean(scores)
        print(f"Параметры: {params}, Средний R^2: {mean_score}")

        # Сравнение и сохранение лучших параметров
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    print(f"\nЛучшая комбинация: {best_params} с R^2: {best_score}")


comparison_tree()

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import joblib

# Пути для сохранения/загрузки данных. Не меняйте эти пути.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "result"
DT_MODEL_PATH = RESULTS_DIR / "decision_tree_model.pkl"
RF_MODEL_PATH = RESULTS_DIR / "random_forest_model.pkl"
GB_MODEL_PATH = RESULTS_DIR / "gradient_boosting_model.pkl"
SCALER_PATH = RESULTS_DIR / "feature_scaler.pkl"
TRAIN_DATASET_PATH = DATA_DIR / "train_data.npz"


def save_object(model, filename):
    """
    Сохраняет модель в файл. Не меняйте эту функцию.
    
    Аргументы:
        model: Обученная модель
        filename: Путь для сохранения модели
    """
    joblib.dump(model, filename)


def load_data():
    """
    Загружает тренировочные данные из директории данных. Эта функция уже рабочая, так что лучше ее не трогать.
    
    Возвращает:
        tuple: (X_train, y_train) где X_train - матрица признаков для обучения, 
               а y_train - метки классов для обучения
    """
    data = np.load(TRAIN_DATASET_PATH)
    X_train = data['X_train']
    y_train = data['y_train']
    
    return X_train, y_train


def preprocess_data(X, y):
    """
    Обрабатывает данные, применяя масштабирование признаков.
    
    Аргументы:
        X: Матрица признаков
        y: Метки классов
        
    Возвращает:
        tuple: (X_scaled, y, scaler)
    """
    # TODO: Реализуйте предобработку данных
    # 1. Создайте и обучите StandardScaler на данных
    # 2. Примените масштабирование к данным
    # 3. Верните обработанные данные и объект масштабировщика
    
    # Замените этот код своей реализацией
    
    # Создайте объект масштабировщика
    scaler = StandardScaler()  # TODO: Инициализируйте масштабировщик StandardScaler
    
    # Обучите и примените масштабирование к данным
    X_scaled = scaler.fit_transform(X)  # TODO: Обучите и примените scaler к X
    
    return X_scaled, y, scaler


def train_decision_tree(X_train, y_train):
    """
    Обучает простой классификатор на основе дерева решений.
    
    Аргументы:
        X_train: Признаки для обучения
        y_train: Метки классов для обучения
        
    Возвращает:
        object: Обученная модель дерева решений
    """
    # TODO: Реализуйте обучение дерева решений
    # 1. Создайте DecisionTreeClassifier с подходящими гиперпараметрами
    # 2. Обучите модель на тренировочных данных
    # 3. Верните обученную модель
    
    # Замените этот код своей реализацией
    # Рекомендации:
    # - Для предотвращения переобучения рассмотрите параметры max_depth, min_samples_split, min_samples_leaf
    # - Учтите возможную несбалансированность классов с помощью class_weight
    # - Выберите подходящий критерий разбиения (criterion): 'gini' или 'entropy'

    param_grid = {
        'max_depth': [4, 6, 8, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    }
    
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )  # TODO: Инициализируйте классификатор на основе дерева решений
    
    # Обучите модель
    # TODO: Обучите модель
    grid.fit(X_train,y_train)
    print("Лучшие параметры для дерева решений:", grid.best_params_)
    
    return grid.best_estimator_


def train_random_forest(X_train, y_train):
    """
    Обучает классификатор на основе случайного леса.
    
    Аргументы:
        X_train: Признаки для обучения
        y_train: Метки классов для обучения
        
    Возвращает:
        object: Обученная модель случайного леса
    """
    # TODO: Реализуйте обучение случайного леса
    # 1. Создайте RandomForestClassifier с подходящими гиперпараметрами
    # 2. Обучите модель на тренировочных данных
    # 3. Верните обученную модель
    
    # Замените этот код своей реализацией
    # Рекомендации:
    # - Экспериментируйте с количеством деревьев (n_estimators)
    # - Ограничьте глубину деревьев (max_depth) и минимальное количество образцов для разделения (min_samples_split)
    # - Используйте bootstrap=True для обучения каждого дерева на случайной подвыборке
    # - Установите подходящее значение для max_features (обычно 'sqrt' или 'log2')
    # - Учтите возможную несбалансированность классов с помощью class_weight
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'class_weight': [None, 'balanced']
    }
    
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )  # TODO: Инициализируйте классификатор на основе дерева решений
    
    # Обучите модель
    # TODO: Обучите модель
    grid.fit(X_train,y_train)
    print("Лучшие параметры для случайного леса:", grid.best_params_)
    
    return grid.best_estimator_


def train_gradient_boosting(X_train, y_train):
    """
    Обучает классификатор на основе градиентного бустинга с использованием CatBoost.
    
    Аргументы:
        X_train: Признаки для обучения
        y_train: Метки классов для обучения
        
    Возвращает:
        object: Обученная модель градиентного бустинга CatBoost
    """
    # TODO: Реализуйте обучение градиентного бустинга с помощью CatBoost
    # 1. Создайте CatBoostClassifier с подходящими гиперпараметрами
    # 2. Обучите модель на тренировочных данных
    # 3. Верните обученную модель
    
    # Замените этот код своей реализацией
    # Рекомендации:
    # - Подберите оптимальное количество деревьев (iterations)
    # - Экспериментируйте со скоростью обучения (learning_rate)
    # - Ограничьте глубину деревьев (depth) для предотвращения переобучения
    # - Используйте параметр auto_class_weights для обработки несбалансированных классов
    # - Настройте параметр l2_leaf_reg для регуляризации
    # - Для ускорения обучения можно использовать параметр task_type='GPU', если доступен GPU

    model = CatBoostClassifier(
        verbose = 0,
        random_state=42
    )
    
    param_grid = {
        'iterations': [100, 200],
        'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5],
        'auto_class_weights': ['Balanced']
    }
    
    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=1
    )  # TODO: Инициализируйте классификатор на основе дерева решений
    
    # Обучите модель
    # TODO: Обучите модель
    grid.fit(X_train,y_train)
    print("Лучшие параметры для CatBoost:", grid.best_params_)
    
    return grid.best_estimator_


def evaluate_model(model, X, y, model_name):
    """
    Оценивает производительность модели на данных.
    
    Аргументы:
        model: Обученная модель
        X: Признаки для оценки
        y: Истинные метки классов
        model_name: Название модели для вывода
        
    Возвращает:
        tuple: (accuracy, f1, roc_auc) - точность, F1-мера и ROC-AUC
    """
    # TODO: Реализуйте оценку модели
    # 1. Выполните предсказания классов и вероятностей с помощью модели
    # 2. Рассчитайте метрики производительности
    
    # Замените этот код своей реализацией
    y_pred = model.predict(X)  # TODO: Сделайте предсказания классов с помощью model.predict()

    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:,1]
    else:
        y_proba = y_pred  # TODO: Получите вероятности для класса 1 с помощью model.predict_proba()[:, 1]
    
    # Рассчитайте метрики
    accuracy = accuracy_score(y,y_pred)  # TODO: Рассчитайте accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)# TODO: Рассчитайте f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)  # TODO: Рассчитайте roc_auc_score(y, y_proba)
    
    print(f"Метрики для модели {model_name}:")
    print(f"Точность: {accuracy:.4f}")
    print(f"F1-мера: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    return accuracy, f1, roc_auc


def main():
    """
    Основная функция для выполнения рабочего процесса:
    1. Загрузка тренировочных данных
    2. Предобработка данных
    3. Обучение разных моделей на основе деревьев решений
    4. Оценка всех моделей на валидационных данных
    5. Сохранение всех моделей и масштабировщика
    """
    print("Код начал выполняться")
    # Создаем директории для данных и результатов, если они не существуют
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Загрузка тренировочных данных
    X_train, y_train = load_data()
    print(f"Загружены тренировочные данные: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Анализ распределения классов
    class_counts = np.bincount(y_train)
    print(f"Распределение классов в тренировочном наборе: {class_counts}")
    print(f"Доля класса 1: {class_counts[1] / len(y_train):.4f}")
    
    # Предобработка данных
    X_train_scaled, y_train, scaler = preprocess_data(X_train, y_train)
    print(f"Данные обработаны: X_train_scaled={X_train_scaled.shape}")
    
    # Разделение данных на обучающую и валидационную выборки
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    print(f"Разделение на обучающую и валидационную выборки:")
    print(f"X_train_split={X_train_split.shape}, y_train_split={y_train_split.shape}")
    print(f"X_val={X_val.shape}, y_val={y_val.shape}")
    
    # Обучение разных моделей
    print("Обучение моделей...\n")
    
    # Обучение и оценка дерева решений
    print("Обучение модели дерева решений...")
    dt_model = train_decision_tree(X_train_split, y_train_split)
    print("Оценка модели дерева решений на валидационных данных...")
    dt_metrics = evaluate_model(dt_model, X_val, y_val, "дерево решений")
    
    # Обучение и оценка случайного леса
    print("\nОбучение модели случайного леса...")
    rf_model = train_random_forest(X_train_split, y_train_split)
    print("Оценка модели случайного леса на валидационных данных...")
    rf_metrics = evaluate_model(rf_model, X_val, y_val, "случайный лес")
    
    # Обучение и оценка градиентного бустинга
    print("\nОбучение модели градиентного бустинга...")
    gb_model = train_gradient_boosting(X_train_split, y_train_split)
    print("Оценка модели градиентного бустинга на валидационных данных...")
    gb_metrics = evaluate_model(gb_model, X_val, y_val, "градиентный бустинг")
    
    # Вывод сравнения моделей
    print("\nСравнение моделей на валидационных данных:")
    print("Модель            | Точность | F1-мера | ROC-AUC")
    print("--------------------|----------|---------|--------")
    print(f"Дерево решений     | {dt_metrics[0]:.4f}   | {dt_metrics[1]:.4f}  | {dt_metrics[2]:.4f}")
    print(f"Случайный лес      | {rf_metrics[0]:.4f}   | {rf_metrics[1]:.4f}  | {rf_metrics[2]:.4f}")
    print(f"Градиентный бустинг| {gb_metrics[0]:.4f}   | {gb_metrics[1]:.4f}  | {gb_metrics[2]:.4f}")
    
    # Обучение финальных моделей на всех тренировочных данных
    print("\nОбучение финальных моделей на всех тренировочных данных...")
    
    final_dt_model = train_decision_tree(X_train_scaled, y_train)
    final_rf_model = train_random_forest(X_train_scaled, y_train)
    final_gb_model = train_gradient_boosting(X_train_scaled, y_train)
    
    # Сохранение моделей и масштабировщика
    print("Сохранение моделей и масштабировщика...")
    save_object(final_dt_model, DT_MODEL_PATH)
    save_object(final_rf_model, RF_MODEL_PATH)
    save_object(final_gb_model, GB_MODEL_PATH)
    save_object(scaler, SCALER_PATH)
    
    print(f"Модель дерева решений сохранена в {DT_MODEL_PATH}")
    print(f"Модель случайного леса сохранена в {RF_MODEL_PATH}")
    print(f"Модель градиентного бустинга сохранена в {GB_MODEL_PATH}")
    print(f"Масштабировщик сохранен в {SCALER_PATH}")
    
    print("\nПримечание: Для прохождения тестов необходимо достичь следующих значений метрик:")
    print("Дерево решений: ROC-AUC >= 0.75")
    print("Случайный лес: ROC-AUC >= 0.90")
    print("Градиентный бустинг: ROC-AUC >= 0.90") 

    if __name__ == "__main__":
        main()

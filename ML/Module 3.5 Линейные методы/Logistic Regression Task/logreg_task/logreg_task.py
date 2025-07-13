import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve
import os
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Пути для сохранения/загрузки данных. Не меняйте эти пути.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "result"
MODEL_PATH = RESULTS_DIR / "best_logreg_model.pkl"
SCALER_PATH = RESULTS_DIR / "feature_scaler.pkl"
THRESHOLD_PATH = RESULTS_DIR / "optimal_threshold.pkl"
TRAIN_DATASET_PATH = DATA_DIR / "train_data.npz"


def save_object(model, filename):
    """
    Сохраняет модель и не только в файл. Не меняйте эту функцию.
    
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


def train_logreg_model(X_train, y_train):
    """
    Обучает классификатор логистической регрессии с оптимальными гиперпараметрами.
    Реализуем поиск лучших гиперпараметров логистической регрессии с помощью GridSearchCV.
    
    Аргументы:
        X_train: Признаки для обучения
        y_train: Метки классов для обучения
        
    Возвращает:
        object: Обученная модель логистической регрессии
    """
    # TODO: Реализуйте обучение модели логистической регрессии
    # 1. Создайте LogisticRegression с оптимальными гиперпараметрами
    # 2. Обучите модель на тренировочных данных
    # 3. При необходимости используйте кросс-валидацию для поиска оптимальных параметров
    # 4. Верните обученную модель
    
    # Замените этот код своей реализацией
    # Рекомендации:
    # - Для несбалансированных данных рассмотрите параметр class_weight='balanced'
    # - Экспериментируйте с разными значениями C (например, 0.01, 0.1, 1.0, 10.0)
    # - Попробуйте разные solver (например, 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
    # - Для некоторых solver подходит только определенный penalty ('l2', 'l1', 'elasticnet', None)
    # - При необходимости увеличьте max_iter для обеспечения сходимости

    param_grid = [
    {
        'penalty': ['l1'],
        'solver': ['liblinear', 'saga'],
        'C': [0.00001,0.0001,0.01, 0.1, 1.0, 10,100,500,1000],
    },
    {
        'penalty': ['l2'],
        'solver': ['liblinear', 'lbfgs', 'saga'],
        'C': [0.00001,0.0001,0.01, 0.1, 1.0, 10,100,500,1000],
    },
    {
        'penalty': ['elasticnet'],
        'solver': ['saga'],
        'l1_ratio': [0.3, 0.5, 0.7],
        'C': [0.00001,0.0001,0.01, 0.1, 1.0, 10,100,500,1000],
    }
]

    logreg = LogisticRegression(
        class_weight='balanced',
        max_iter=10000
    )  # TODO: Инициализируйте классификатор логистической регрессии с оптимальными параметрами
    
    grid_search = GridSearchCV(
        estimator=logreg,
        param_grid=param_grid,
        scoring='average_precision',
        cv=10,
        n_jobs=-1,
        verbose=1
    )
    # Обучите модель
    # TODO: Обучите модель

    grid_search.fit(X_train,y_train)

    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучший средний precision: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def get_optimal_threshold(model, X, y):
    """
    Возвращает оптимальный порог для классификации, обеспечивающий recall >= 0.8
    с максимальной precision.
    
    Аргументы:
        model: Обученная модель
        X: Признаки для оценки
        y: Истинные метки классов
        
    Возвращает:
        float: Оптимальный порог для классификации
    """
    # TODO: Реализуйте функцию поиска оптимального порога
    # 1. Получите вероятности для класса 1 с помощью model.predict_proba(X)[:, 1]
    # 2. Вычислите precision и recall для различных порогов с помощью precision_recall_curve
    # 3. Найдите индексы порогов, где recall >= 0.8
    # 4. Выберите индекс с максимальной precision среди валидных индексов
    # 5. Верните оптимальный порог
    
    # Замените этот код своей реализацией
    y_scores = model.predict_proba(X)[:,1]
    precisions, recalls, thresholds = precision_recall_curve(y,y_scores)
    valid_idx = [i for i,r in enumerate(recalls) if r>=0.8]

    if not valid_idx:
        print("Не найдено порогов с recall >= 0.8")
        return 0.5
    
    best_idx = max(valid_idx,key = lambda i : precisions[i])
    optimal_threshold = thresholds[best_idx]  # TODO: Реализуйте поиск оптимального порога
    
    print(f"Выбранный порог: {optimal_threshold:.4f} (Recall: {recalls[best_idx]:.4f}, Precision: {precisions[best_idx]:.4f})")

    return optimal_threshold



def evaluate_model(model, X, y):
    """
    Оценивает производительность модели на данных.
    
    Аргументы:
        model: Обученная модель
        X: Признаки для оценки
        y: Истинные метки классов
        
    Возвращает:
        tuple: (accuracy, average_precision) - точность и средняя точность
    """
    # TODO: Реализуйте оценку модели
    # 1. Выполните предсказания с помощью модели
    # 2. Рассчитайте метрики производительности
    
    # Замените этот код своей реализацией
    y_pred = model.predict(X)  # TODO: Сделайте предсказания классов с помощью model.predict()
    y_proba = model.predict_proba(X)[:,1]  # TODO: Получите вероятности для класса 1 с помощью model.predict_proba()[:, 1]
    
    # Рассчитайте метрики
    accuracy = accuracy_score(y,y_pred)  # TODO: Рассчитайте accuracy_score(y, y_pred)
    avg_precision = average_precision_score(y,y_proba)  # TODO: Рассчитайте average_precision_score(y, y_proba)
    
    print(f"Точность модели: {accuracy:.4f}")
    print(f"Average Precision модели: {avg_precision:.4f}")
    
    return accuracy, avg_precision


def main():
    """
    Основная функция для выполнения рабочего процесса логистической регрессии:
    1. Загрузка тренировочных данных
    2. Предобработка данных
    3. Обучение модели логистической регрессии
    4. Оценка модели с использованием Average Precision
    5. Нахождение оптимального порога для классификации
    6. Сохранение модели, масштабировщика и оптимального порога
    """
    # Создаем директории для данных и результатов, если они не существуют
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Загрузка тренировочных данных
    X_train, y_train = load_data()
    print(f"Загружены тренировочные данные: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Анализ распределения классов
    class_counts = np.bincount(y_train)
    print(f"Распределение классов в тренировочном наборе: {class_counts}")
    print(f"Доля класса 1 (миноритарного): {class_counts[1] / len(y_train):.4f}")
    
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
    
    # Обучение модели логистической регрессии
    print("Обучение модели логистической регрессии...")
    logreg_model = train_logreg_model(X_train_split, y_train_split)
    
    # Оценка модели на валидационной выборке
    print("Оценка модели на валидационной выборке...")
    accuracy, avg_precision = evaluate_model(logreg_model, X_val, y_val)
    
    # Нахождение оптимального порога для классификации
    print("Поиск оптимального порога для достижения recall >= 0.8 с максимальной precision...")
    optimal_threshold = get_optimal_threshold(logreg_model, X_val, y_val)
    
    # Обучение финальной модели на всех тренировочных данных
    print("Обучение финальной модели на всех тренировочных данных...")
    final_model = train_logreg_model(X_train_scaled, y_train)
    
    # Нахождение оптимального порога для финальной модели
    print("Поиск оптимального порога для финальной модели...")
    final_optimal_threshold = get_optimal_threshold(final_model, X_train_scaled, y_train)
    
    # Сохранение модели, масштабировщика и оптимального порога
    print("Сохранение модели, масштабировщика и оптимального порога...")
    save_object(final_model, MODEL_PATH)
    save_object(scaler, SCALER_PATH)
    save_object(final_optimal_threshold, THRESHOLD_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")
    print(f"Масштабировщик сохранен в {SCALER_PATH}")
    print(f"Оптимальный порог сохранен в {THRESHOLD_PATH}")
    print("\nПримечание: Для прохождения теста Average Precision должен быть не менее 0.888299")


if __name__ == "__main__":
    main() 
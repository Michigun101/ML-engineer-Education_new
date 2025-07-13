import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
from pathlib import Path
import joblib

# Пути для сохранения/загрузки данных. Не меняйте эти пути.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "result"
MODEL_PATH = RESULTS_DIR / "best_knn_model.pkl"
SCALER_PATH = RESULTS_DIR / "feature_scaler.pkl"
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
    scaler = StandardScaler()  # TODO: Инициализируйте масштабировщик
    
    # Обучите и примените масштабирование к данным
    X_scaled = scaler.fit_transform(X)  # TODO: Обучите и примените scaler к X
    
    return X_scaled, y, scaler


def train_knn_model(X_train, y_train):
    """
    Обучает классификатор KNN с оптимальными гиперпараметрами.
    
    Аргументы:
        X_train: Признаки для обучения
        y_train: Метки классов для обучения
        
    Возвращает:
        object: Обученная модель KNN
    """
    # TODO: Реализуйте обучение модели KNN
    # 1. Создайте KNeighborsClassifier с оптимальными гиперпараметрами
    # 2. Обучите модель на тренировочных данных
    # 3. Примените кросс-валидацию для поиска оптимальных параметров
    # 4. Верните обученную модель
    
    # Замените этот код своей реализацией
    # Экспериментируйте с разными значениями n_neighbors, weights и metric
    # чтобы найти лучшую комбинацию
    param_grid = {
        'n_neighbors': list(range(3, 16)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    knn = KNeighborsClassifier()  # TODO: Инициализируйте классификатор KNN с оптимальными параметрами
    grid_search = GridSearchCV(knn,param_grid,cv=5,scoring='accuracy',n_jobs=-1)
    
    
    # Обучите модель
    # TODO: Обучите модель
    grid_search.fit(X_train,y_train)
    print("Лучшие параметры:", grid_search.best_params_)
    print("Лучшая точность на кросс-валидации:", grid_search.best_score_)
    
    return grid_search.best_estimator_


def evaluate_model(model, X, y):
    """
    Оценивает производительность модели на данных.
    
    Аргументы:
        model: Обученная модель
        X: Признаки для оценки
        y: Истинные метки классов
        
    Возвращает:
        float: Точность классификации
    """
    # TODO: Реализуйте оценку модели
    # 1. Выполните предсказания с помощью модели
    # 2. Рассчитайте и верните точность
    
    # Замените этот код своей реализацией
    y_pred = model.predict(X)  # TODO: Сделайте предсказания
    accuracy = accuracy_score(y,y_pred)  # TODO: Рассчитайте точность
    
    print(f"Точность модели на данных: {accuracy:.4f}")
    return accuracy


def main():
    """
    Основная функция для выполнения рабочего процесса KNN:
    1. Загрузка тренировочных данных
    2. Предобработка данных
    3. Обучение модели KNN с использованием кросс-валидации
    4. Сохранение модели и масштабировщика
    """
    # Создаем директорию для результатов, если она не существует
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Загрузка тренировочных данных
    X_train, y_train = load_data()
    print(f"Загружены тренировочные данные: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Предобработка данных
    X_train_scaled, y_train, scaler = preprocess_data(X_train, y_train)
    print(f"Данные обработаны: X_train_scaled={X_train_scaled.shape}")
    
    # Обучение модели KNN
    print("Обучение модели KNN с использованием кросс-валидации...")
    knn_model = train_knn_model(X_train_scaled, y_train)
    
    # Оценка модели на тренировочных данных с использованием кросс-валидации
    # Обратите внимание, что у нас нет доступа к тестовым данным
    print("Оценка модели с использованием кросс-валидации...")
    evaluate_model(knn_model,X_train_scaled,y_train)
    
    # TODO: Сохраните модель и масштабировщик с помощью функции save_model
    # Модель должна быть сохранена по пути MODEL_PATH
    # Масштабировщик должен быть сохранен по пути SCALER_PATH
    print("Сохранение модели и масштабировщика...")
    save_object(knn_model, MODEL_PATH)
    save_object(scaler, SCALER_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")
    print(f"Масштабировщик сохранен в {SCALER_PATH}")


if __name__ == "__main__":
    main()

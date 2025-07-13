import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import os
from pathlib import Path
import joblib

# Пути для сохранения/загрузки данных. Не меняйте эти пути.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "result"
PIPELINE_PATH = RESULTS_DIR / "model_pipeline.pkl"
TRAIN_DATASET_PATH = DATA_DIR / "train_data.npz"


def save_object(model, filename):
    """
    Сохраняет модель и другие объекты в файл.
    
    Аргументы:
        model: Обученная модель или другой объект
        filename: Путь для сохранения объекта
    """
    joblib.dump(model, filename)


def load_data():
    """
    Загружает тренировочные данные из директории данных.
    
    Возвращает:
        tuple: (X_train, y_train) где X_train - матрица признаков для обучения, 
               а y_train - целевые значения для обучения
    """
    data = np.load(TRAIN_DATASET_PATH)
    X_train = data['X_train']
    y_train = data['y_train']
    
    return X_train, y_train


def create_pipeline(degree = 3):
    """
    Создает и настраивает пайплайн для трансформации данных и обучения модели.
    
    Возвращает:
        object: Настроенный пайплайн sklearn
    """
    # TODO: Реализуйте создание и настройку пайплайна
    # 1. Создайте пайплайн с необходимыми шагами трансформации и моделью
    # 2. Задайте оптимальные параметры для каждого шага
    # 3. Верните настроенный пайплайн
    
    # Замените этот код своей реализацией
    
    
    # Пример базового пайплайна
    pipeline = Pipeline([
        ('scaler',StandardScaler()),
        ('poly',PolynomialFeatures(degree=degree,include_bias=False)),
        ('regression',LinearRegression())
    ])  # TODO: Создайте пайплайн с необходимыми шагами трансформации
    
    return pipeline


def train_model(X_train, y_train):
    """
    Создает и обучает пайплайн на тренировочных данных.
    
    Аргументы:
        X_train: Признаки для обучения
        y_train: Целевые значения для обучения
        
    Возвращает:
        object: Обученный пайплайн
    """
    # TODO: Реализуйте создание и обучение пайплайна
    # 1. Создайте пайплайн с помощью функции create_pipeline
    # 2. Обучите пайплайн на тренировочных данных
    # 3. Верните обученный пайплайн
    
    # Замените этот код своей реализацией
    
    pipeline = create_pipeline(degree=3) # degree = 2,3,4
    
    # Обучите пайплайн
    # TODO: Обучите пайплайн на тренировочных данных
    pipeline.fit(X_train,y_train)
    
    return pipeline


def evaluate_model(model, X, y):
    """
    Оценивает производительность модели на данных.
    
    Аргументы:
        model: Обученный пайплайн
        X: Признаки для оценки
        y: Истинные целевые значения
        
    Возвращает:
        tuple: (r2, mse) - коэффициент детерминации и среднеквадратичная ошибка
    """
    # TODO: Реализуйте оценку модели
    # 1. Выполните предсказания с помощью пайплайна
    # 2. Рассчитайте метрики производительности
    
    # Замените этот код своей реализацией
    y_pred = model.predict(X)  # TODO: Сделайте предсказания с помощью model.predict()
    
    # Рассчитайте метрики
    r2 = r2_score(y,y_pred)  # TODO: Рассчитайте r2_score(y, y_pred)
    mse = mean_squared_error(y,y_pred)  # TODO: Рассчитайте mean_squared_error(y, y_pred)
    
    print(f"R² модели: {r2:.4f}")
    print(f"MSE модели: {mse:.2f}")
    
    return r2, mse


def main():
    """
    Основная функция для выполнения рабочего процесса линейной регрессии:
    1. Загрузка тренировочных данных
    2. Создание и обучение пайплайна, включающего предобработку данных и модель
    3. Оценка модели
    4. Сохранение пайплайна
    """
    # Создаем директории для данных и результатов, если они не существуют
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Загрузка тренировочных данных
    X_train, y_train = load_data()
    print(f"Загружены тренировочные данные: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Разделение данных на обучающую и валидационную выборки
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    print(f"Разделение на обучающую и валидационную выборки:")
    print(f"X_train_split={X_train_split.shape}, y_train_split={y_train_split.shape}")
    print(f"X_val={X_val.shape}, y_val={y_val.shape}")
    
    # Обучение модели на тренировочной выборке
    print("Обучение модели...")
    pipeline = train_model(X_train_split, y_train_split)
    
    # Оценка модели на валидационной выборке
    print("Оценка модели на валидационной выборке...")
    r2_val, mse_val = evaluate_model(pipeline, X_val, y_val)
    
    # Обучение финальной модели на всех тренировочных данных
    print("Обучение финальной модели на всех тренировочных данных...")
    final_pipeline = train_model(X_train, y_train)
    
    # Оценка финальной модели на тренировочных данных
    print("Оценка финальной модели на тренировочных данных...")
    r2_train, mse_train = evaluate_model(final_pipeline, X_train, y_train)
    
    # Сохранение пайплайна
    print("Сохранение пайплайна...")
    save_object(final_pipeline, PIPELINE_PATH)
    print(f"Пайплайн сохранен в {PIPELINE_PATH}")
    print("\nПримечание: Для прохождения теста R² должен быть не менее 0.9489")


if __name__ == "__main__":
    main() 
# Задание по алгоритмам на основе деревьев решений

## Описание задания

В этом задании вам предстоит реализовать и сравнить различные алгоритмы машинного обучения на основе деревьев решений:
1. Дерево решений (Decision Tree)
2. Случайный лес (Random Forest)
3. Градиентный бустинг (CatBoost)

Вам необходимо реализовать предобработку данных, обучение и настройку всех трех моделей, а также оценку их производительности. Каждая модель будет проверяться отдельно с соответствующими ожиданиями по качеству.


## Задачи для выполнения

Вам необходимо заполнить пропущенные части кода в файле `decision_trees_task/decision_trees_task.py`:

1. `preprocess_data` - функция предобработки данных с использованием StandardScaler
2. `train_decision_tree` - функция обучения модели на основе дерева решений
3. `train_random_forest` - функция обучения модели на основе случайного леса
4. `train_gradient_boosting` - функция обучения модели на основе градиентного бустинга (CatBoost)
5. `evaluate_model` - функция оценки производительности модели

В каждой функции есть комментарии с рекомендациями по реализации.

## Данные

Тренировочные данные загружаются автоматически из файла `data/train_data.npz` с помощью функции `load_data()`. Данные представляют собой бинарную задачу классификации со сложной структурой, где различные алгоритмы могут показать разную производительность.


## Критерии оценки

Ваше решение будет оцениваться отдельно для каждой модели на основе достижения определенных пороговых значений метрик:

**Дерево решений:**
- Accuracy (точность): не менее 0.70
- F1-мера: не менее 0.68
- ROC-AUC: не менее 0.75

**Случайный лес:**
- Accuracy (точность): не менее 0.84
- F1-мера: не менее 0.75
- ROC-AUC: не менее 0.90

**Градиентный бустинг (CatBoost):**
- Accuracy (точность): не менее 0.86
- F1-мера: не менее 0.80
- ROC-AUC: не менее 0.92


## Советы

- **Для дерева решений:**
  - Подберите оптимальную глубину дерева и другие параметры для предотвращения переобучения
  - Дерево решений может не справиться с некоторыми сложными нелинейными связями в данных

- **Для случайного леса:**
  - Экспериментируйте с количеством деревьев и параметрами случайности
  - Обычно случайный лес показывает более стабильные результаты по сравнению с одиночным деревом решений

- **Для градиентного бустинга (CatBoost):**
  - Подберите правильный баланс между количеством итераций и скоростью обучения
  - Используйте параметры регуляризации для предотвращения переобучения
  - CatBoost обычно дает лучшие результаты на многих задачах по сравнению с другими реализациями градиентного бустинга
  - Для больших наборов данных можно использовать GPU для ускорения обучения

- Обратите внимание на возможную несбалансированность классов
- Используйте кросс-валидацию для надежной оценки моделей
- В данных присутствуют нелинейные связи, которые могут быть сложны для простого дерева решений

# -*- coding: utf-8 -*-
"""
Скрипт обучения ML модели с интеграцией MLflow
Использует датасет Iris для классификации
"""

import os
import sys

# ВАЖНО: Установка переменных окружения ДО импорта MLflow
# Используем абсолютный путь к текущей рабочей директории
WORK_DIR = os.path.abspath(os.getcwd())
MLRUNS_PATH = 'mlruns'

# Создаем директорию mlruns заранее, чтобы MLflow не пытался создавать её в другом месте
os.makedirs(MLRUNS_PATH, exist_ok=True)

# В CI/CD окружении переопределяем HOME и другие директории
if os.getenv('CI') == 'true' or os.getenv('GITHUB_ACTIONS') == 'true':
    os.environ['HOME'] = WORK_DIR
    os.environ['USERPROFILE'] = WORK_DIR  # Для Windows
    os.environ['HOMEDRIVE'] = ''
    os.environ['HOMEPATH'] = WORK_DIR
    print(f"CI/CD обнаружен, HOME установлен на: {WORK_DIR}")

# Устанавливаем все переменные окружения MLflow
MLRUNS_ABS_PATH = os.path.abspath(MLRUNS_PATH)
os.environ['MLFLOW_TRACKING_URI'] = f'file://{MLRUNS_ABS_PATH}'
os.environ['MLFLOW_ARTIFACT_ROOT'] = MLRUNS_ABS_PATH
os.environ['MLFLOW_REGISTRY_URI'] = f'file://{MLRUNS_ABS_PATH}'
os.environ['MLFLOW_ARTIFACT_LOCATION'] = MLRUNS_ABS_PATH
os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = MLRUNS_ABS_PATH

# Запрещаем MLflow использовать любые системные директории
os.environ['MLFLOW_EXPERIMENT_NAME'] = 'iris_classification'
os.environ['MLFLOW_RUN_ID'] = ''

print(f"MLflow будет использовать: {MLRUNS_ABS_PATH}")

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime

def create_directories():
    """Создание необходимых директорий для проекта"""
    directories = ['models', 'data', 'reports', 'mlruns']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Директории созданы успешно")

def load_and_prepare_data():
    """Загрузка и подготовка данных Iris"""
    print("Загрузка датасета Iris...")
    iris = load_iris()
    
    # Создание DataFrame для удобства работы
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['target'] = iris.target
    
    # Сохранение данных для дальнейшего использования
    df.to_csv('data/iris_dataset.csv', index=False)
    print(f"Датасет сохранен в data/iris_dataset.csv")
    print(f"Размер датасета: {df.shape}")
    
    return df, iris

def train_model(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=5, random_state=42):
    """Обучение модели Random Forest с логированием в MLflow"""
    
    # Явно устанавливаем tracking URI перед каждым экспериментом
    mlruns_path = 'mlruns'
    mlruns_abs = os.path.abspath(mlruns_path)
    
    # Убеждаемся, что директория существует
    os.makedirs(mlruns_abs, exist_ok=True)
    
    # Устанавливаем tracking URI с явным протоколом file://
    tracking_uri = f'file://{mlruns_abs}'
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Текущая рабочая директория: {os.getcwd()}")
    print(f"HOME: {os.environ.get('HOME', 'не установлен')}")
    
    # Установка имени эксперимента
    mlflow.set_experiment("iris_classification")
    
    with mlflow.start_run():
        print("\nНачало обучения модели...")
        
        # Логирование параметров
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state,
            'criterion': 'gini',
            'min_samples_split': 2
        }
        mlflow.log_params(params)
        
        # Создание и обучение модели
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Расчет метрик
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, average='weighted')
        recall = recall_score(y_test, y_pred_test, average='weighted')
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        # Логирование метрик
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        mlflow.log_metrics(metrics)
        
        # Вывод результатов
        print(f"\nРезультаты обучения:")
        print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
        print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        
        # Сохранение модели
        model_path = 'models/random_forest_model.joblib'
        joblib.dump(model, model_path)
        print(f"\nМодель сохранена в {model_path}")
        
        # Логирование модели в MLflow (с обработкой ошибок для CI/CD)
        try:
            # Получаем информацию о текущем run ДО логирования модели
            run_id = mlflow.active_run().info.run_id
            experiment_id = mlflow.active_run().info.experiment_id
            mlflow_uri = mlflow.get_tracking_uri()
            artifact_path = mlflow.active_run().info.artifact_uri
            
            print(f"\nИнформация о MLflow run:")
            print(f"  Tracking URI: {mlflow_uri}")
            print(f"  Artifact URI: {artifact_path}")
            print(f"  Experiment ID: {experiment_id}")
            print(f"  Run ID: {run_id}")
            
            # Логируем модель и артефакт
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(model_path)
            
            print(f"\nМодель успешно залогирована в MLflow")
            print(f"Локальное сохранение: {os.path.abspath(model_path)}")
            print(f"MLflow директория: {os.path.abspath(mlruns_path)}")
            print(f"MLflow артефакты: {os.path.abspath(mlruns_path)}/{experiment_id}/{run_id}/artifacts/")
        except Exception as e:
            import traceback
            print(f"\nПредупреждение: не удалось залогировать модель в MLflow: {e}")
            print(f"Тип ошибки: {type(e).__name__}")
            print(f"Полный traceback:")
            traceback.print_exc()
            print(f"\nДиагностическая информация:")
            print(f"  MLflow tracking URI: {mlflow.get_tracking_uri()}")
            print(f"  Текущая директория: {os.getcwd()}")
            print(f"  HOME: {os.environ.get('HOME', 'не установлен')}")
            print(f"  MLFLOW_TRACKING_URI: {os.environ.get('MLFLOW_TRACKING_URI', 'не установлен')}")
            print("Модель сохранена локально, продолжаем работу...")
        
        # Логирование дополнительной информации
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset", "Iris")
        mlflow.set_tag("training_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        return model, metrics

def main():
    """Основная функция для запуска обучения"""
    print("=" * 60)
    print("Запуск ML пайплайна с MLflow")
    print("=" * 60)
    
    # Создание директорий
    create_directories()
    
    # Загрузка данных
    df, iris = load_and_prepare_data()
    
    # Разделение на признаки и целевую переменную
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nРазмер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    
    # Обучение модели
    model, metrics = train_model(X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 60)
    print("Обучение завершено успешно")
    print("=" * 60)
    
    return model, metrics

if __name__ == "__main__":
    main()

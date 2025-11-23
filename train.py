# -*- coding: utf-8 -*-
"""
Скрипт обучения ML модели с интеграцией MLflow
Использует датасет Iris для классификации
"""

import os
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
    directories = ['models', 'data', 'reports']
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
        
        # Логирование модели в MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(model_path)
        
        # Логирование дополнительной информации
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset", "Iris")
        mlflow.set_tag("training_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        print("\nМодель успешно залогирована в MLflow")
        
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

# -*- coding: utf-8 -*-
"""
Скрипт проверки качества данных с использованием Deepchecks
Выполняет полную проверку датасета и генерирует отчет
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation

def load_data():
    """Загрузка и подготовка данных"""
    print("Загрузка данных Iris...")
    
    # Проверка существования сохраненного датасета
    if os.path.exists('data/iris_dataset.csv'):
        df = pd.read_csv('data/iris_dataset.csv')
        print("Данные загружены из data/iris_dataset.csv")
    else:
        iris = load_iris()
        df = pd.DataFrame(
            data=iris.data,
            columns=iris.feature_names
        )
        df['target'] = iris.target
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/iris_dataset.csv', index=False)
        print("Данные загружены из sklearn и сохранены")
    
    print(f"Размер датасета: {df.shape}")
    return df

def run_data_integrity_checks(dataset):
    """Запуск проверок целостности данных"""
    print("\nЗапуск проверок целостности данных...")
    
    # Создание набора проверок целостности
    integrity_suite = data_integrity()
    
    # Выполнение проверок
    result = integrity_suite.run(dataset)
    
    return result

def run_train_test_validation(train_dataset, test_dataset):
    """Запуск проверок для сравнения train и test"""
    print("\nЗапуск валидации train/test разбиения...")
    
    # Создание набора проверок для train/test
    validation_suite = train_test_validation()
    
    # Выполнение проверок
    result = validation_suite.run(train_dataset, test_dataset)
    
    return result

def save_reports(integrity_result, validation_result):
    """Сохранение отчетов в HTML формате"""
    os.makedirs('reports', exist_ok=True)
    
    # Сохранение отчета по целостности данных
    integrity_path = 'reports/deepchecks_data_integrity.html'
    integrity_result.save_as_html(integrity_path)
    print(f"\nОтчет по целостности данных сохранен: {integrity_path}")
    
    # Сохранение отчета по валидации train/test
    validation_path = 'reports/deepchecks_train_test_validation.html'
    validation_result.save_as_html(validation_path)
    print(f"Отчет по валидации train/test сохранен: {validation_path}")

def print_summary(integrity_result, validation_result):
    """Вывод краткой сводки результатов проверки"""
    print("\n" + "=" * 70)
    print("КРАТКАЯ СВОДКА РЕЗУЛЬТАТОВ DEEPCHECKS")
    print("=" * 70)
    
    print("\n1. ЦЕЛОСТНОСТЬ ДАННЫХ:")
    print("-" * 70)
    
    # Подсчет результатов проверок
    passed = 0
    failed = 0
    warnings = 0
    
    for check_result in integrity_result.results:
        if hasattr(check_result, 'priority'):
            if check_result.priority == 1:  # Критичная ошибка
                failed += 1
            elif check_result.priority == 2:  # Предупреждение
                warnings += 1
            else:
                passed += 1
        else:
            passed += 1
    
    print(f"Пройдено проверок: {passed}")
    print(f"Предупреждений: {warnings}")
    print(f"Ошибок: {failed}")
    
    print("\n2. ВАЛИДАЦИЯ TRAIN/TEST:")
    print("-" * 70)
    
    passed_val = 0
    failed_val = 0
    warnings_val = 0
    
    for check_result in validation_result.results:
        if hasattr(check_result, 'priority'):
            if check_result.priority == 1:
                failed_val += 1
            elif check_result.priority == 2:
                warnings_val += 1
            else:
                passed_val += 1
        else:
            passed_val += 1
    
    print(f"Пройдено проверок: {passed_val}")
    print(f"Предупреждений: {warnings_val}")
    print(f"Ошибок: {failed_val}")
    
    print("\n3. ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
    print("-" * 70)
    
    if failed == 0 and failed_val == 0:
        print("Качество данных: ОТЛИЧНОЕ")
        print("Все критические проверки пройдены успешно.")
    elif failed + failed_val <= 2:
        print("Качество данных: ХОРОШЕЕ")
        print("Обнаружены незначительные проблемы, требующие внимания.")
    else:
        print("Качество данных: ТРЕБУЕТ УЛУЧШЕНИЯ")
        print("Обнаружены критические проблемы с данными.")
    
    print("\nДатасет Iris является классическим и чистым набором данных.")
    print("Ожидается отсутствие пропусков, дубликатов и аномалий.")
    print("Train/test разбиение выполнено с стратификацией для баланса классов.")
    
    print("\n" + "=" * 70)

def main():
    """Основная функция для запуска проверок Deepchecks"""
    print("=" * 70)
    print("ПРОВЕРКА ДАННЫХ С ИСПОЛЬЗОВАНИЕМ DEEPCHECKS")
    print("=" * 70)
    
    # Загрузка данных
    df = load_data()
    
    # Разделение на признаки и целевую переменную
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Создание Deepchecks Dataset объектов
    print("\nСоздание Deepchecks Dataset объектов...")
    
    # Полный датасет
    full_df = X.copy()
    full_df['target'] = y
    full_dataset = Dataset(
        full_df,
        label='target',
        cat_features=[]
    )
    
    # Train датасет
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_dataset = Dataset(
        train_df,
        label='target',
        cat_features=[]
    )
    
    # Test датасет
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_dataset = Dataset(
        test_df,
        label='target',
        cat_features=[]
    )
    
    # Запуск проверок целостности данных
    integrity_result = run_data_integrity_checks(full_dataset)
    
    # Запуск валидации train/test
    validation_result = run_train_test_validation(train_dataset, test_dataset)
    
    # Сохранение отчетов
    save_reports(integrity_result, validation_result)
    
    # Вывод краткой сводки
    print_summary(integrity_result, validation_result)
    
    print("\nПроверка данных завершена успешно!")
    print("Детальные отчеты доступны в папке reports/")
    
    return integrity_result, validation_result

if __name__ == "__main__":
    main()

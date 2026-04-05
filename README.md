# Phishing Email Detection

ML-система для определения фишинговых писем с использованием алгоритмов машинного обучения.

## Структура проекта

```
phishing-detection/
├── data/                       # Данные
│   ├── raw/                    # Исходные датасеты
│   ├── processed/              # Очищенные данные
│   └── features/               # Извлечённые признаки
├── notebooks/                  # Jupyter Notebooks
│   ├── 01_eda.ipynb            # EDA анализ
│   ├── 02_feature_engineering.ipynb # Извлечение признаков
│   └── 03_model_training.ipynb # Обучение моделей
├── src/                        # Исходный код
│   ├── features/               # Извлечение признаков
│   │   ├── sender_features.py  # Признаки отправителя
│   │   ├── url_features.py    # Признаки URL
│   │   ├── text_features.py    # Текстовые признаки
│   │   └── attachment_features.py # Признаки вложений
│   ├── models/                 # Обучение моделей
│   │   ├── train_xgboost.py
│   │   └── train_logistic_regression.py
│   └── utils/                  # Утилиты
│       ├── preprocessing.py
│       └── evaluation.py
├── models/                     # Сохранённые модели
├── app/                        # Веб-приложение
│   ├── app.py
│   └── templates/
├── tests/                      # Тесты
├── config.yaml                 # Конфигурация
└── requirements.txt            # Зависимости
```

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### 1. Анализ данных и извлечение признаков

Откройте ноутбуки в `notebooks/`:
- `01_eda.ipynb` — анализ данных
- `02_feature_engineering.ipynb` — извлечение признаков
- `03_model_training.ipynb` — обучение моделей

### 2. Обучение моделей

```bash
python src/models/train_xgboost.py
python src/models/train_logistic_regression.py
```

### 3. Запуск веб-приложения

```bash
python app/app.py
```

Откройте http://localhost:8000 в браузере.

## Датасет

Используется CEAS_08 dataset с меткой `label`:
- `0` — легитимное письмо
- `1` — фишинговое письмо

## Модели

- **XGBoost** — градиентный бустинг (основная модель)
- **Logistic Regression** — линейная модель для сравнения

## Результаты

| Метрика | XGBoost | Logistic Regression |
|---------|---------|---------------------|
| Accuracy | 99.41% | 94.85% |
| Precision | 99.40% | 94.37% |
| Recall | 99.54% | 96.48% |
| F1-score | 99.47% | 95.41% |
| ROC-AUC | 99.97% | 98.08% |

## Признаки

- **TF-IDF**: 5000 n-грамм (unigrams + bigrams)
- **Ручные признаки** (24 шт.):
  - Признаки отправителя: домен, бесплатный email, длина домена
  - Признаки URL: количество, длина, подозрительные TLD, IP-адреса
  - Текстовые признаки: триггерные слова (urgency, threat, reward), заглавные буквы, знаки препинания
  - Признаки вложений: подозрительные расширения, макросы

## Тесты

```bash
pytest tests/
```

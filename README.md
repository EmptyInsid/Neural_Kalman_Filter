# Neural_Kalman_Filter

Проект, посвящённый исследованию возможностей фильтра Каламана и его улучшения с помощью нейронных сетей

---

## Directories

```
Neural_Kalman_Filter/
├── experiment/          # Папка для экспериментов и тестов
│   ├── audio_base/       # эксперименты с применением фильтра для аудио
│   ├── classic/          # реализация классического фильтра Калмана на python
│   └── matlab_calman/    # реализация классического фильтра Калмана на matlab
└── app/                 # Основная папка с вариантами приложения
    ├── classic_calman_app/      # Основная папака с приложением
    |   ├── gui/              # GUI для приложения
    |   └── model/            # Функции и модели для Фильтра Калмана
    └── neural_calman_app/       # Основная папака с приложением
        ├── gui/              # GUI для приложения
        ├── neural/           # Модели нейронных сетей для фильтра
        └── model/            # Функции и модели для Фильтра Калмана
```

## Quick start

Предварительные условия:
- python3 >= 3.11.2

```
# Установка зависимостей и создание venv
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Запуск экспериментальных скриптов
python3 -m experiment.classic.by_matlab

# Запуск классического приложения
python3 -m app.classic_calman_app.gui.gui

# Запуск приложения с нейросетью
python3 -m app.neural_calman_app.neural.train_model # предварительно необходимо обучить модель
python3 -m app.neural_calman_app.gui.gui

```

# neuroflow_terminal
Интерактивный терминал (REPL) сделает работу с NeuroFlow ещё более удобной.
Полноценная интерактивную консоль с подсветкой синтаксиса, автодополнением, историей команд и визуализацией модели в реальном времени.

Структура проекта

neuroflow_terminal/

├── repl.py         # Главный файл REP

├── lexer.py                   # Лексический анализатор

├── parser.py                  # Синтаксический анализатор

├── interpreter.py             # Интерпретатор кода

├── visualizer.py              # ASCII визуализация

├── completer.py               # Автодополнение

├── history.py                 # История команд
└── requirements.txt

1. requirements.txt
requirements.txt

prompt_toolkit==3.0.43
pygments==2.17.0
torch==2.1.0
numpy==1.24.0

2. Запуск терминала

# 1. Создайте папку проекта
mkdir neuroflow_terminal
cd neuroflow_terminal

# 2. Сохраните все файлы выше

# 3. Установите зависимости
pip install -r requirements.txt

# 4. Запустите терминал
python repl.py

8. Пример сессии

╔══════════════════════════════════════════════════════════════╗

║           🧠 NeuroFlow Interactive Terminal v1.0             ║

╚══════════════════════════════════════════════════════════════╝

neuroflow> input image [28, 28]

✅ Code executed successfully

neuroflow> input label [10]

✅ Code executed successfully

neuroflow> model MyNet {
...     x = flatten(image)
...     h1 = dense(x, 128) -> relu
...     out = dense(h1, 10) -> softmax
... }

✅ Model 'MyNet' defined with 3 layers

╔══════════════════════════════════════════════════════════════╗

║  Model: MyNet                                                ║

║  INPUTS:                                                     ║

║    📥 image               [28, 28]                           ║

║    📥 label               [10]                               ║

║  LAYERS:                                                     ║

║    1. 📏 flatten          [flatten]                          ║

║    2. 🔵 h1               [dense]                            ║

║    3. 🔵 out              [dense]                            ║

║  TOTAL PARAMETERS: 101,642                                   ║

╚══════════════════════════════════════════════════════════════╝

neuroflow> /layers

📊 Layer Statistics:
============================================================
  flatten              | Params: 0         | Shape: (784,) → (784,)
  h1                   | Params: 100,480   | Shape: (784,) → (128,)
  out                  | Params: 1,290     | Shape: (128,) → (10,)
============================================================

neuroflow> /save my_model.nf

✅ Session saved to my_model.nf

neuroflow> /exit

👋 Goodbye!

9. Возможности терминала

  Фича	Описание
  
**Подсветка синтаксиса**	Цветные ключевые слова, функции, числа

**Автодополнение**	Tab-комплит для команд и функций

**История команд**	↑/↓ для навигации по истории

**Multi-line режим**	Блоки кода с автоматическим продолжением

**ASCII визуализация**	Мгновенный просмотр архитектуры

**Специальные команды**	/help, /model, /save, /load, etc.

**Сохранение сессий**	Экспорт/импорт кода в .nf файлы

**Статистика слоёв**	Подсчёт параметров и размеров



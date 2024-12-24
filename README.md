# Presentation-RAG

Проект для семантического поиска по PDF-презентациям.

В бизнесе и науке информацию часто предоставляют в виде презентаций: обзоры отрасли, квартальные отчеты, рекламные рассылки, выступления на конференциях. Если презентаций много, то нужен инструмент для поиска по ним. При поиске по презентациям хочется учитывать текстовое и визуальное содержимое:  списки, таблицы, схемы, картинки, графики.

Идея проекта: с помощью image2text моделей составить описания слайдов, которые учитывают текстовое и визуальное содержимое. А затем искать по ним с помощью векторной базы данных.

Автор: Илья Тамбовцев

Попробовать можно тут: [huggingface/presentation-search](https://huggingface.co/spaces/redmelonberry/presentation-search)

## Функционал

- Извлечение структурированной информации из слайдов презентаций с помощью GPT-4V
- Векторное хранилище описаний слайдов на базе ChromaDB
- Семантический поиск по презентациям с учетом текстового и визуального контента
- Веб-интерфейс для удобного доступа к функциональности поиска

## Демонстрация
**Здесь будет видео?гифка?скрины?**

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repo_url>
cd presentation-rag
```

2. Установите зависимости через poetry:
```bash
poetry install
```

3. Настройте переменные окружения в `.env`:
```
# Use OpenAI
OPENAI_API_KEY=""

# Or you can use vsegpt.ru
VSEGPT_API_BASE="https://api.vsegpt.ru/v1"
VSEGPT_API_KEY=""

# Setup Langsmith Evaluation
LANGSMITH_API_KEY=""
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_PROJECT="presentation_rag"
LANGCHAIN_TRACING_V2="true"

# Configure custom benchmark spreadsheet
BENCHMARK_SPREADSHEET_ID="1qWRF_o-RY1x-o-3z08iVb2akh0HS3ZNxVkZi6yoVsI4"

# Configure ChromaDB storage name for webapp
CHROMA_COLLECTION_NAME="pres_69"
```

## Использование

### 0. Сбор данных
Добавьте презентации в папку raw. Структура директории не важна.

Скачать мои презентации можно [по ссылке](https://drive.google.com/drive/folders/1nUBF-MZgSyuyQEemUXtCj1VDoe1NSqBa?usp=drive_link)

### 1. Анализ презентаций

Запустите скрипт для анализа презентаций. Достаточно указать названия файлов без расширения. Не обязательно указывать полный путь.

В результате в папке `interim` появятся json'ы с описаниями презентаций.

```bash
poetry run python -m src.run_descriptions process \
    "4.Обзор уязвимостей и техник защиты для LLM_Евгений Кокуйкин_вер.3" \
    "6.От промпта к агентной системе_Никита Венедиктов_вер.2" \
    "5. Козлов Жизнь в платформе.pptx" \
    \
    --provider=vsegpt \
    --model-name="vis-openai/gpt-4o-mini" \
    --max-concurrent=5
```


### 2. Создание векторной базы ChromaDB

Преобразуйте результаты анализа в векторное хранилище.

В результате в папке `processed` появится локальное хранилище ChromaDB.

```bash
poetry run python -m src.run_json2chroma convert \
    "4.Обзор уязвимостей и техник защиты для LLM_Евгений Кокуйкин_вер.3" \
    "6.От промпта к агентной системе_Никита Венедиктов_вер.2" \
    "5. Козлов Жизнь в платформе.pptx" \
    \
    --collection=pres1 \
    --mode=fresh \
    --provider=openai
```

### 3. Веб-интерфейс

Запустите веб-приложение:

```bash
poetry run python -m src.webapp.app --collection pres1 --host 0.0.0.0 --port 7860
```

### Бенчмарк
#### MLFlow
Запустите MLFlow для отображения результатов:

```sh
poetry run mlflow ui --backend-store-uri sqlite:///data/processed/eval/runs/mlruns.db
```

Запустите тестирование:
```sh
poetry run python -m src.run_evaluation mlflow \
    --retriever="basic" \
    --n_contexts=-1 \
    --n_pages=-1 \
    --provider="openai" \
    --scorers=[basic] \
    --metrics=[basic] \
    --max_concurrent=10 \
    --model_name="gpt-4o-mini" \
    --collection="pres0" \
    --experiment="PresRetrieve_25" \
    --n_questions=-1 \
    --temperature=0.2 \
    --sheet_id="Q15" \
    --write_to_google=false \
    --rate_limit_timeout=-1
```

##### Результаты Бенчмаркинга
![картинка](docs/img/MlFlow_Eval_69_Preprocessors_short.png)

В датасете 69 презентаций. Каждый эксперимент отвечает отдельному методу реранкинга. Есть две группы - с regex-препроцессингом и без препроцессинга.
Первые два столбца в экспериментах одинаковые, потому что retriever у них один.
Разница в том, в каком порядке они будут выданы пользователю. Хочется сделать такой реранкинг, чтобы выдача была более релевантной. Подробнее про реранкинг [есть ноутбук](notebooks/rag/chroma_metric_research.ipynb)

Интерпретация:
В 90% мы находим нужную презентацию (`presentationfound`). При этом в среднем входит в топ-2 или топ-3 выдачи (`presentationidx`). Однако на первом месте в выдаче нужная презентация оказывается в ~70% случаев (`presentationmatch`).

Выводы:
- с препроцессингом лучше
- если использовать препроцессинг, то лучше всего работает обычный `min-scorer` - без реранкинга
- если без препроцессинга, то лучше всех отработал `hyperbolic-scorer`

Еще я пробовал оценивать LLM-кой релевантность выдачи. Просил давать оценку релевантности от 0-10. Промпт можно посмотреть [тут](src/eval/eval_mlflow.py):

Получилось неинформативно - при большой разнице в `match` разница по оценке LLM маленькая.

![llm-relevance](docs/img/MLFlow_LLM-Eval.png)

#### Langsmith
**Здесь будет гайд как запустить бенчмаркинг через LangSmith**


## Ноутбуки

**Вкатываемся в Document Understanding**

1. [Ноутбук с примерами описаний слайдов](notebooks/weird-slides/weird_slides.ipynb)
2. [Ноутбук с примерами LLamaParse](notebooks/weird-slides/llamaparse.ipynb)
3. [Ноутбук с ресерчем про разрешения картинки](notebooks/weird-slides/lowering_img_quality.ipynb)
4. [Ноутбук с разрешениями презентаций и расчетом цен](notebooks/data_description/count_descriptions_costs.ipynb)

**Описания презентаций**

5. [Ноутбук с примерами форматированных описаний](notebooks/prompts/testing_prompts.ipynb)
6. [Ноутбук с распределением количества токенов в полученных описаниях](notebooks/data_description/count_descriptions_metrics.ipynb)

**RAG**

7. [Ноутбук с примерами запросов к RAG](notebooks/rag/chroma_queries.ipynb)
8. [Ноутбук про реранкинг](notebooks/rag/chroma_metric_research.ipynb)

## Структура проекта

```
presentation-rag/
├── data/              # Директория данных
│   ├── raw/           # Исходные презентации
│   ├── interim/       # Промежуточные результаты анализа
│   └── processed/     # Обработанные данные и векторные базы
├── docs/              # Документация
├── notebooks/         # Jupyter ноутбуки
└── src/               # Исходный код
    ├── chains/        # Цепочки обработки с LangChain
    ├── config/        # Конфигурация
    ├── eda/           # Разведочный анализ данных
    ├── eval/          # Оценка качества
    ├── processing/    # Утилиты обработки
    ├── rag/           # Модули RAG
    ├── testing_utils/ # Тестовые утилиты
    └── webapp/        # Веб-приложение
```
### Основные модули (src/)

#### chains/
Цепочки обработки на базе LangChain для анализа презентаций:
- `chains.py` - Базовые цепочки для работы с PDF и изображениями (FindPdfChain, LoadPageChain, Page2ImageChain и др.)
- `pipelines.py` - Пайплайны для обработки презентаций (SingleSlidePipeline, PresentationPipeline)
- `prompts.py` - Промпты для анализа слайдов (JsonH1AndGDPrompt)

#### config/
Конфигурация и утилиты проекта:
- `config.py` - Основной класс конфигурации
- `model_setup.py` - Настройка языковых моделей и эмбеддингов
- `navigator.py` - Навигация по структуре проекта
- `output_formatting.py` - Форматирование вывода
- `logging.py` - Настройка логирования

#### eda/
Инструменты для разведочного анализа данных:
- `explore.py` - Функции для анализа презентаций, подсчета токенов и оценки стоимости

Есть ноутбуки:
- [анализ pdf-файлов презентаций](notebooks/data_description/count_presentations_metrics.ipynb)
- [анализ описаний от gpt-vision](notebooks/data_description/count_descriptions.ipynb)

#### eval/
Модули для оценки качества:
- `eval_mlflow.py` - Метрики и логирование экспериментов в MLflow
- `evaluate.py` - Оценка качества результатов с использованием LangSmith

#### processing/
Утилиты для обработки файлов:
- `image_utils.py` - Работа с изображениями (конвертация в base64)
- `pdf_utils.py` - Работа с PDF (конвертация страниц в изображения)

#### rag/
Компоненты для Retrieval-системы:

`preprocess.py` - Предобработка поисковых запросов.

В проекте используется regex-препроцессинг - убираются часто-повторяющиеся фразы. Идея: "а сильно по-другому к презентации вопрос и не задашь" - [см. бенчмарк из 200 вопросов](https://docs.google.com/spreadsheets/d/1qWRF_o-RY1x-o-3z08iVb2akh0HS3ZNxVkZi6yoVsI4/edit?usp=sharing)

`score.py` - Скоринг результатов поиска для реранкинга

Идея: "Давайте делать презентацию более релевантной, если поиск вернул много страниц из нее". [Подробнее есть ноутбук](notebooks/rag/chroma_metric_research.ipynb)

`storage.py` - Работа с векторным хранилищем ChromaDB

Два режима - с препроцессингом, и без. [Есть ноутбук с примерами запросов к базе](notebooks/rag/chroma_queries.ipynb)

#### testing_utils/
Утилиты для тестирования:
- `echo_llm.py` - Тестовая LLM, возвращающая входные данные

#### webapp/
Веб-интерфейс на Gradio:
- `app.py` - Основное веб-приложение для поиска по презентациям

### Основные скрипты

- `run_descriptions.py` - Запуск анализа презентаций с помощью GPT-4V
- `run_json2chroma.py` - Конвертация результатов анализа в векторное хранилище
- `run_evaluation.py` - Бенчмаркинг поисковика
- `run_webapp.py` - Запуск приложения на gradio

## Дополнительная документация

- [System Design Document](docs/system_design_doc.md)
- [Гайд по DVC](docs/workflow/data_version_control.md)
- [Гайд по разрешениям картинок](docs/workflow/resolutions.md)

## Ссылки

### Данные
- [Гугл-диск с презентациями](https://drive.google.com/drive/folders/1IvUsxxtyyTuHdZff9szhd2OtIATRTcG4?usp=sharing)
- [Гугл-таблица с бенчмарком](https://docs.google.com/spreadsheets/d/1qWRF_o-RY1x-o-3z08iVb2akh0HS3ZNxVkZi6yoVsI4/edit?usp=sharing)


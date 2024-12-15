# Presentation-RAG

Проект для семантического поиска по PDF-презентациям.

В бизнесе и науке информацию часто предоставляют в виде презентаций: обзоры отрасли, квартальные отчеты, рекламные рассылки, выступления на конференциях. Если презентаций много, то нужен инструмент для поиска по ним. При поиске по презентациям хочется учитывать текстовое и визуальное содержимое:  списки, таблицы, схемы, картинки, графики.

Идея проекта: с помощью image2text моделей составить описания слайдов, которые учитывают текстовое и визуальное содержимое. А затем искать по ним с помощью векторной базы данных.

Автор: Илья Тамбовцев

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
OPENAI_API_KEY="sk-OGTI_AZVY2pZTGCj6_ZnzZojCx9YDThDm4SueiGTcYT3BlbkFJu9t0OHQM_Q7S3bDt8shTpNuYL_0qdpR_wVZG8VIeUA"

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
```

## Использование

### 0. Сбор данных
Добавьте презентации в папку raw. Структура директории не важна.

### 1. Анализ презентаций

Запустите скрипт для анализа презентаций. Достаточно указать названия файлов без расширения. Не обязательно указывать полный путь.

В результате в папке `interim` появятся json'ы с описаниями презентаций.

```bash
poetry run python -m src.run_descriptions process \
    presentation_1 \
    presentation_2 \
    \
    --provider vsegpt \
    --model-name vis-openai/gpt-4o-mini \
    --max-concurrent 3
```


### 2. Создание базы знаний

Преобразуйте результаты анализа в векторное хранилище.

В результате в папке `processed` появится локальное хранилище ChromaDB.

```bash
poetry run python -m src.run_json2chroma convert \
    presentation_1 \
    presentation_2 \
    \
    --collection pres1 \
    --mode fresh \
    --provider openai
```

### 3. Веб-интерфейс

Запустите веб-приложение:

```bash
poetry run python -m src.webapp.app --collection pres1 --host 0.0.0.0 --port 7860
```

### Бенчмарк
**Здесь будет гайд как запустить бенчмаркинг**


## Ноутбуки

**Вкатываемся в Document Understanding**

1. [Ноутбук с примерами описаний слайдов](https://gitlab.raftds.com/ilia.tambovtsev/presentation-rag/-/blob/ee6268ca210a9721d2f247251349aca43517f68d/notebooks/weird-slides/weird_slides.ipynb)
2. [Ноутбук с примерами LLamaParse](https://gitlab.raftds.com/ilia.tambovtsev/presentation-rag/-/blob/ee6268ca210a9721d2f247251349aca43517f68d/notebooks/weird-slides/llamaparse.ipynb)
3. [Ноутбук с ресерчем про разрешения картинки](https://gitlab.raftds.com/ilia.tambovtsev/presentation-rag/-/blob/ee6268ca210a9721d2f247251349aca43517f68d/notebooks/weird-slides/lowering_img_quality.ipynb)
4. [Ноутбук с разрешениями презентаций и расчетом цен](https://gitlab.raftds.com/ilia.tambovtsev/presentation-rag/-/blob/ee6268ca210a9721d2f247251349aca43517f68d/notebooks/weird-slides/lowering_img_quality.ipynb)

**Описания презентаций**

5. [Ноутбук с примерами форматированных описаний](https://gitlab.raftds.com/ilia.tambovtsev/presentation-rag/-/blob/ee6268ca210a9721d2f247251349aca43517f68d/notebooks/prompts/testing_prompts.ipynb)
6. [Ноутбук с распределением количества токенов в полученных описаниях](https://gitlab.raftds.com/ilia.tambovtsev/presentation-rag/-/blob/a71ffa99b9a223440210918f5bbcfff91aa040a5/notebooks/data_description/count_descriptions.ipynb)

**RAG**

7. [Ноутбук с примерами запросов к RAG](https://gitlab.raftds.com/ilia.tambovtsev/presentation-rag/-/blob/ee6268ca210a9721d2f247251349aca43517f68d/notebooks/rag/chroma_queries.ipynb)
8. [Ноутбук про реранкинг](https://gitlab.raftds.com/ilia.tambovtsev/presentation-rag/-/blob/514f171fb260faec3f2d2580fa78fe9313fc17ee/notebooks/rag/chroma_metric_research.ipynb)

## Структура проекта

```
presentation-rag/
├── data/               # Директория данных
│   ├── raw/           # Исходные презентации
│   ├── interim/       # Промежуточные результаты анализа
│   └── processed/     # Обработанные данные и векторные базы
├── docs/              # Документация
├── notebooks/         # Jupyter ноутбуки
└── src/              # Исходный код
    ├── chains/       # Цепочки обработки с LangChain
    ├── config/       # Конфигурация
    ├── eda/          # Разведочный анализ данных
    ├── eval/         # Оценка качества
    ├── processing/   # Утилиты обработки
    ├── rag/          # Модули RAG
    ├── testing_utils/# Тестовые утилиты
    └── webapp/       # Веб-приложение
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

#### eval/
Модули для оценки качества:
- `eval_mlflow.py` - Метрики и логирование экспериментов в MLflow
- `evaluate.py` - Оценка качества результатов с использованием LangSmith

#### processing/
Утилиты для обработки файлов:
- `image_utils.py` - Работа с изображениями (конвертация в base64)
- `pdf_utils.py` - Работа с PDF (конвертация страниц в изображения)

#### rag/
Компоненты для Retrieval Augmented Generation:
- `preprocess.py` - Предобработка поисковых запросов
- `score.py` - Скоринг результатов поиска
- `storage.py` - Работа с векторным хранилищем ChromaDB

#### testing_utils/
Утилиты для тестирования:
- `echo_llm.py` - Тестовая LLM, возвращающая входные данные

#### webapp/
Веб-интерфейс на Gradio:
- `app.py` - Основное веб-приложение для поиска по презентациям

### Основные скрипты

- `run_descriptions.py` - Запуск анализа презентаций с помощью GPT-4V
- `run_json2chroma.py` - Конвертация результатов анализа в векторное хранилище

## Дополнительная документация

- [System Design Document](docs/system_design_doc.md)
- [Data Version Control Guide](docs/workflow/data_version_control.md)
- [API Documentation](docs/api.md)

## Ссылки

### Данные
- [Гугл-диск с презентациями](https://drive.google.com/drive/folders/1IvUsxxtyyTuHdZff9szhd2OtIATRTcG4?usp=sharing)
- [Гугл-таблица с бенчмарком](https://docs.google.com/spreadsheets/d/1qWRF_o-RY1x-o-3z08iVb2akh0HS3ZNxVkZi6yoVsI4/edit?usp=sharing)
  - [Результаты бенчмарка в Langsmith]() # TODO

### Проект
- [Jira Board](https://jira.raftds.com/secure/RapidBoard.jspa?rapidView=98&projectKey=RMI&selectedIssue=RMI-40)
- [Confluence Wiki](https://confluence.raftds.com/display/RMI1/Raft+ML+internship+1)
- [GitLab Repository](https://gitlab.raftds.com/ilia.tambovtsev/presentation-rag)
- [Google Drive - Presentations](https://drive.google.com/drive/folders/1IvUsxxtyyTuHdZff9szhd2OtIATRTcG4?usp=sharing)

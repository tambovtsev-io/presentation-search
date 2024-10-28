# Presentation-RAG System Design

## Цели и предпосылки

### Зачем идем в разработку продукта?

**Цель проекта:** Разработать эффективный инструмент для автоматизированного поиска информации по базе из презентаций.

**Проблематика:**
- Большой объем бизнес-информации хранится в форме презентаций.
- Поиск по ним усложняется визуальным содержимым - графиками, диаграммами, картинками.
- Нет универсального подхода - решение задачи зависит от домена информации и визуального содержимого
- `Сложность навигации по докладам на многосекционных конференциях`

**Почему станет лучше от использования ML:**
1. LLM способны интерпретировать естественно-языковые запросы и связывать их с релевантным контентом.
2. Visual Transformers (ViT) эффективно анализируют содержимое изображений и распознают текст.
3. Интеграция ViT и LLM улучшает понимание контекста документов.
4. RAG позволяет эффективно работать с неструктурированными данными, такими как презентации.

### Ожидаемые результаты
#### Пользовательский опыт
- Пользователь вводит естественно-языковой запрос в поисковую строку.
- Ожидание в течение *адекватного* времени.
- Результат:
  - Ранжированный список из релевантных презентаций/слайдов.
  - Ссылки на оригиналы.
  - Можно просматривать презентации.
- Если результат не устроил, можно задать переформулированный запрос в том же чате.
- История поиска сохраняется.


Сценарий бизнес:
1. Подготовка к встрече:
   - Пользователь ищет "последние презентации о финансовых показателях компании"
   - Система выдает релевантные слайды из различных презентаций
   - Пользователь быстро составляет краткий отчет на основе найденной информации

2. Анализ конкурентов:
   - Запрос: "Сравнение наших продуктов с конкурентами за последний квартал"
   - Система находит слайды с таблицами сравнения и диаграммами
   - Пользователь легко идентифицирует ключевые различия и преимущества

3. Подготовка презентации:
   - Поиск "креативные слайды о запуске нового продукта"
   - Система предлагает разнообразные визуальные решения из прошлых презентаций
   - Пользователь быстро адаптирует найденные идеи для новой презентации

Сценарий исследователь:
1. Анализ трендов:
   - Запрос: "Графики роста рынка AI за последние 5 лет"
   - Система выдает релевантные графики из различных презентаций
   - Исследователь анализирует тренды и формирует гипотезы

2. Поиск методологии:
   - Пользователь ищет "методология оценки эффективности ML моделей"
   - Система находит слайды с описанием методов и формулами
   - Исследователь изучает различные подходы и выбирает наиболее подходящий

3. Сравнение результатов:
   - Запрос: "Сравнительные таблицы производительности GPU и TPU"
   - Система предоставляет релевантные таблицы и графики
   - Исследователь сопоставляет данные для своего эксперимента


#### Функционал поиска
##### Обработка запросов
- Поддержка запросов на естественном языке, включающих текстовое и визуальное содержание.
  - Интерпретация запроса и понимание намерения пользователя.
  - Обработка запросов со сложной струтурой:
    - Несколько условий
    - ~~Произвольное форматирование: списки, **bold**, CAPS~~
    - Произвольная формулировка:
      - Вопросы, требования, их комбинации
    - Термины?
  - Поиск по визуальными элементами презентаций.
    - Стиль (минималистичный, "много текста", ...)
    - Диаграммы разных типов (круговые, столбчатые, графики)
    - Табличные данные
    - Метаданные (автор, дата, ...)
    - Специфические визуальные элементы (воронки продаж, таймлайны, схемы, ...)
- Примеры поддерживаемых запросов:
  - "Найди презентации в минималистичном стиле о стратегии компании"
  - "Покажи слайды с воронкой продаж в презентациях о digital-маркетинге"
  - "Выдай презентацию по data-аналитике, в которой были картинки с лягушками"
  - "В каких презентациях были графики зависимости X от размера модели?"

##### Генерация ответов
- Запрос пользователя проходит предобработку для поиска. Формулируется запрос для RAG
- RAG выдает ссылки на релевантные результаты
- К результам генерируется пояснение в контексте запроса. Ответ на вопрос "Почему этот слайд попал в ответ?"

#### Веб интерфейс
- Чат-интерфейс
- Поле для поиска
- Панель с результатами. Можно экспортировать ссылки.
- Можно задавать сколько угодно вопросов. История сохраняется.
- Чат сохраняется


## Методология
### Постановка задачи
Разработка *ассистента* на основе LLM и RAG для автоматического поиска информации из *набора* презентаций.

%%
- На какую тему презентации? Какой домен? Важно ли это? 
- Что делать с терминами? Если они нетипичные?
- Как загружаются презы? Это сервис с двумя типами акков: хост и юзер?
- Как измерять качество поиска?
- Как размечаем датасет?
- Объем?
- Позитивные/негативные примеры
- Что делаем с шумом? В презентации часто добавляют картинки для отвлечения внимания. Хотя они же могут и запомниться
- Мб обучать темам по одним данным а обучать поиску по другим?
%%

- Данные - [Подробное описание](data/description.md)
  - Презентации про ML на русском языке (100)
    - AIConf
    - YappiDays
  - Презентации на бизнес тематику (100): отчеты, обзоры, ...
    - [tg: businessincognita](https://t.me/businessincognita)
    - [tg: insider_infor](https://t.me/insider_infor)


### Этапы
#### Proof of Concept (PoC)
##### Данные
- 10 простых презентаций
  - до 30 слайдов ?
  - много текста
  - простой дизайн: текст, картинки четко отделены от текста (границы у графиков, контраст с фоном)

##### Вопросы

| Доля | О чем?                                | Что в презентации?                             | Примеры                                                                                                          |
| ---- | ------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 60%  | Строго по тексту (заголовки, контент) | Подробный текст про X                          | Найди презентацию на тему `Чтото с титульного слайда` <> В какой презентации рассказывали про `чтото из текста`? |
| 20%  | По обобщенному содержимомоу           | Подробный текст <> Перечисления                | В какой презентации о маркетинге говорили о продажах? <> ...                                                     |
| 10%  | Простые визуальные элементы           | графики <> картинки (четко отделены от текста) | В какой презентации был график зависимости X от Y? <> В какой презентации были изображения машин?                |
| 5%   | Сложные визуальные элементы           | Воронки продаж <> Флоу-чарты                   | Где показывали воронку продаж для X <> ...                                                                       |
| 5%   | Отсылки на визуальные элементы        | Графики продаж <> Диаграмы                     | В какой презентации говорили о продажах за последние 5 лет?                                                      |

Общие моменты
- Вопросы понятные. Понятно намерение. Легко интерпретировать ответ.
- Ответ на вопрос **точно есть** в презентациях. ==Но можно добавить несколько out of domain==
- Ручная проверка.

##### Оформление
Интерфейс - оформление в виде кода / ноутбука. Таблица с отчетом о результатах.

##### Результаты
Что мы узнаем?
- возможности модели
- темп разработки

Сценарий 1: все работает через 2 недели. Тогда идем в усложнение.

Сценарий 2: ничего не работает через месяц. Тогда паника.

По итогам этапа станет понятно, как скорректировать требования для MVP.

#### Minimal Viable Product (MVP)
- 100 презентаций на МЛ тематику. Сложность определим после POC
- Реализованы основные функции поиска
- Веб интерфейс

#### План работы

- Ресерч и тестирование существующих решений `1нед`
  - Что уже есть?
  - Как оценивать качество?
  - Выбор моделей
  - Как разрабатывать?
- Разработка POC
  - Код `1w+`
    - setup `1 день`
      - Удобная архитектура проекта
      - Зависимости: Библиотеки / Окружения / docker / ...
    - Разработка `1нед`
      - Предобработка презентаций
        - Разработка промптов для LLM
      - RAG
        - узнать как пользоваться
        - ...
  - Подготовка небольшого датасета `2 дня`
    - Выбор презентаций `2-3ч`
        - Просмотр
        - Заметки по презентации. Научиться в них ориентироваться. Знать все про эти 10 презентаций
    - Составление вопросов
      - Ресерч как сейчас составляют вопросы `2ч`
      - Составление по шаблону + свои `2ч`
    - Интеграция с RAG - разметка, `1 день`
  - Тестирование
    - Выбор метрик - "как автоматически понимать что выдача ок?"
    - Проведение тестов `1 день на тест`
    - Обсуждения со взрослыми
    - Корректировки, итерации
- Дальнейшее планирование
  - что можно сделать? что нет?
  - будет понятно, как масштабировать
- Разработка MVP
  - Подготовка большого датасета
  - Тестирование
  - Оценка метрик
  - Выводы
  - Разработка веб-интерфейса
    - Ресерч фреймворков `2д`
    - Реализация `3д`
    - Тестирование функционала `1д` -- будет +/- параллельно с другими задачами
    - Фичи
      - Форматы сохранение результатов
      - Юзабилити

##### Времязатраты

**POC**
| Задача                                     | Ожидаемое время | Затраченное время |
| ------------------------------------------ | --------------- | ----------------- |
| Ресерч и тестирование существующих решений | 1н+             |                   |
| Подготовка данных для POC                  | 2д              |                   |
| Разработка POC                             | 1н+             |                   |
| Тестирование и итерации                    | 3д              |                   |


**MVP**
| Задача                                     | Ожидаемое время | Затраченное время | Комментарий                           |
| ------------------------------------------ | --------------- | ----------------- | ------------------------------------- |
| Осмысление результатов POC <> Планирование | 2д              |                   |                                       |
| Подготовка данных для MVP                  | 5д              |                   | как составить много хороших вопросов? |
| Разработка MVP                             | ?               |                   | База есть с POC, что дальше?          |
| Тестирование и итерации                    | 3д              |                   |                                       |
| Веб-интерфейс и деплой                     | 5д              |                   |                                       |


## Студенческие моменты
- Цели
  - Научиться адаптировать RAG для нестандартных задач
  - Презентовать свои результаты перед публикой
  - Закрыть курс проектной практики
- Ожидаемый результат
  - *RAG для поиска по презентациям*
  - Выступление на DataFest
  - Закрытая сессия
tree -L 3 data/raw/data-science/aiconf/День1
: '
data/raw/data-science/aiconf/День1
├── 1.Сфера
│   ├── 1.За рамками сценария_Эмели Драль_вер.3.pdf
│   ├── 2.Kolmogorov Arnold Networks_Павел Плюснин_вер.2.pdf
│   ├── 3.MERA.Text.v.1.2.0. Что под капотом нового релиза_Алена Феногенова_вер.2.pdf
│   ├── 4.Обзор уязвимостей и техник защиты для LLM_Евгений Кокуйкин_вер.3.pdf
│   ├── 5.Устройство и перспективы использования ML-компиляторов_Виталий Шутов_вер.1.p
│   └── 6.Как ML помогает производить лекарства_Владислав Маслов, Василий Вологдин_вер
├── 2.Полусфера
│   ├── 1.Визуальные языковые модели от разбора архитектуры до запуска_Эмиль Шакиров_в
│   ├── 2.Виртуальная примерка одежды для кабинета мерчанта_Роман Лисов_вер.4.pdf
│   ├── 3.Система управления процессом окомкования железорудных окатышей_Андрей Голов_
│   ├── 4.Эволюция отбора кандидатов в системе товарных рекомендаций Ozon_Александр Кр
│   ├── 5.Вместо зеленого экрана_Елизавета Петрова_вер.4.pdf
│   ├── 6.LLM говорит мультимодальные задачи в речевом домене_Борис Жестков_вер.3.pdf
│   └── 7.Генеративные модели для работы с кодом_Евгений Колесников_вер.1.pdf
├── 3.Аудитория 1
│   ├── 1.Превращаем нейросети в SOTA и для табличных задач_Олег Сидоршин_вер.3.pdf
│   ├── 2.Less is more_Дарья Никанорова_вер.3.pdf
│   ├── 3.AutoFE-сапер_Валерия Дымбицкая_вер.3.pdf
│   ├── 4.Как мы делаем прагматичный поиск и Q&A без LLM_Вадим Захаров_вер.6.pdf
│   ├── 5.Фреймворк прикладных инструментов для анализа данных_Ирина Деева_вер.3.pdf
│   ├── 6.Увеличиваем число обнаружений в задачах_Дмитрий Колесников_вер.4.pdf
│   └── 7.Мягкая модерация изображений скрыть нельзя блокировать_Юрий Батраков_вер.4.p
└── 4.Пресс-центр
    ├── 1.Разметка датасетов в эпоху мультимодальности_Дмитрий Антипов_вер.4.pdf
    ├── 2. МК «Получение высококачественных данных для обучения ИИ-моделей»_Олег Секач
    ├── 3.Разметка будущего как GPT помогает обучать модели_Герман Ганус_вер.4.pdf
    ├── 4.Люди не нужны_Данила Бочарников_вер.3.pdf
    └── 5.Жестовый язык особенности сбора данных, опыт и результаты_Петр Суровцев_вер.
'



# Not run
poetry run python -m src.chains.run_descriptions process \
      "1.За рамками сценария_Эмели Драль_вер.3.pdf" \
      "2.Kolmogorov Arnold Networks_Павел Плюснин_вер.2.pdf" \
      "3.MERA.Text.v.1.2.0. Что под капотом нового релиза_Алена Феногенова_вер.2.pdf" \
      "4.Обзор уязвимостей и техник защиты для LLM_Евгений Кокуйкин_вер.3.pdf" \
      "5.Устройство и перспективы использования ML-компиляторов_Виталий Шутов_вер.1.pdf" \
      "6.Как ML помогает производить лекарства_Владислав Маслов, Василий Вологдин_вер.5.pdf" \
      \
      --provider="openai" --max_concurrent=5


poetry run python -m src.chains.run_descriptions process \
"1.Визуальные языковые модели от разбора архитектуры до запуска_Эмиль Шакиров_в" \
"2.Виртуальная примерка одежды для кабинета мерчанта_Роман Лисов_вер.4.pdf" \
"3.Система управления процессом окомкования железорудных окатышей_Андрей Голов_" \
"4.Эволюция отбора кандидатов в системе товарных рекомендаций Ozon_Александр Кр" \
"5.Вместо зеленого экрана_Елизавета Петрова_вер.4.pdf" \
"6.LLM говорит мультимодальные задачи в речевом домене_Борис Жестков_вер.3.pdf" \
"7.Генеративные модели для работы с кодом_Евгений Колесников_вер.1.pdf" \
\
--model="provider" --max_concurrent=5


poetry run python -m src.chains.run_descriptions process \
      "1.Превращаем нейросети в SOTA и для табличных задач_Олег Сидоршин_вер.3.pdf" \
      "2.Less is more_Дарья Никанорова_вер.3.pdf" \
      "3.AutoFE-сапер_Валерия Дымбицкая_вер.3.pdf" \
      "4.Как мы делаем прагматичный поиск и Q&A без LLM_Вадим Захаров_вер.6.pdf" \
      "5.Фреймворк прикладных инструментов для анализа данных_Ирина Деева_вер.3.pdf" \
      "6.Увеличиваем число обнаружений в задачах_Дмитрий Колесников_вер.4.pdf" \
      "7.Мягкая модерация изображений скрыть нельзя блокировать_Юрий Батраков_вер.4.pdf" \
      \
      --provider="openai" --max_concurrent=5


tree -L 3 data/raw/data-science/aiconf/День2
:'
data/raw/data-science/aiconf/День2
├── 1.Сфера
│   ├── 1.Мультимодальные рекомендации в Wildberries_Степан Евстифеев_вер.2.pdf
│   ├── 2.Языковые модели и основы рационального мышления_Ирина Пионтковская_вер.3.pdf
│   ├── 3.3D pose estimation_Aлександр Тимофеев-Каракозов_вер.5.pdf
│   ├── 4.Новый уровень ML-персонализации Lamoda_Дана Злочевская_вер.3.pdf
│   ├── 5.WildBERT — развитие трансформерных архитектур_Евгений Иванов_вер.2.pdf
│   ├── 6.Генерация видео from zero to hero_Денис Димитров_вер.5.pdf
│   └── 7.Где применять LLM, а где это оверкилл_Валентин Малых_вер.3.pdf
├── 2.Полусфера
│   ├── 1.Нейросети в рекомендациях от идеи до продакшна_Любовь Куприянова_вер.7.pdf
│   ├── 2.PostgreSQL для AI_Владлен Пополитов, Олег Бартунов_вер.5.pdf
│   ├── 3.Как «Писец» на «Тотальный диктант» ходил_Иван Бондаренко_вер.2.pdf
│   ├── 4.Побеждают ли диффузионные модели генеративные состязательные сети_Денис Кузнеделев_вер.3.pdf
│   ├── 5.Поиск точек роста ВКонтакте_Степан Малькевич_вер.3.pdf
│   └── 6.Эволюция Transformer как меняется самая успешная архитектура в DL_Мурат Апишев_вер.2.pdf
├── 3.Аудитория 1
│   ├── 1.Что такое ML-платформа на базе K8s_Тимофей Разумов_вер.2.pdf
│   ├── 2.Как AutoML- и AutoDL-сервисы улучшают реальную разработку_Евгений Смирнов_вер.2.pdf
│   ├── 3.Обеспечат ли LLM прорыв в эффективности AutoML_Николай Никитин_вер.2.pdf
│   ├── 4.Где и как использовать LLM в задачах поиска_Валерия Гурьянова_вер.2.pdf
│   ├── 5.Диффузионные модели для мобильных телефонов_Дмитрий Нестеренко_вер.5.pdf
│   ├── 6.От промпта к агентной системе_Никита Венедиктов_вер.2.pdf
│   └── 7.Валидация в RecSys для корреляции с АВ_Дарья Тихонович_вер.7.pdf
└── 4.Пресс-центр
    ├── 1.Физически-обоснованное машинное обучение_Александр Хватов_вер.4.pdf
    ├── 2.МК по работе с геоданными_Артем Каледин, Денис Афанасьев_вер.5.pdf
    ├── 3.ML на графах в e-commerce_Иван Антипов_вер.3.pdf
    ├── 4.Синтетика для поиска редких дефектов_Олег Карташев_вер.3.pdf
    └── 5.Как мы развернули трансформер_Артем Карасюк_вер.3.pdf
'

poetry run python -m src.chains.run_descriptions process \
      "1.Мультимодальные рекомендации в Wildberries_Степан Евстифеев_вер.2.pdf" \
      "2.Языковые модели и основы рационального мышления_Ирина Пионтковская_вер.3.pdf" \
      "3.3D pose estimation_Aлександр Тимофеев-Каракозов_вер.5.pdf" \
      "4.Новый уровень ML-персонализации Lamoda_Дана Злочевская_вер.3.pdf" \
      "5.WildBERT — развитие трансформерных архитектур_Евгений Иванов_вер.2.pdf" \
      "6.Генерация видео from zero to hero_Денис Димитров_вер.5.pdf" \
      "7.Где применять LLM, а где это оверкилл_Валентин Малых_вер.3.pdf" \
      \
      "1.Нейросети в рекомендациях от идеи до продакшна_Любовь Куприянова_вер.7.pdf" \
      "2.PostgreSQL для AI_Владлен Пополитов, Олег Бартунов_вер.5.pdf" \
      "3.Как «Писец» на «Тотальный диктант» ходил_Иван Бондаренко_вер.2.pdf" \
      "4.Побеждают ли диффузионные модели генеративные состязательные сети_Денис Кузнеделев_вер.3.pdf" \
      "5.Поиск точек роста ВКонтакте_Степан Малькевич_вер.3.pdf" \
      "6.Эволюция Transformer как меняется самая успешная архитектура в DL_Мурат Апишев_вер.2.pdf" \
      \
      --provider="openai" --max_concurrent=5

# NOT RUN
poetry run python -m src.chains.run_descriptions process \
      "1.Что такое ML-платформа на базе K8s_Тимофей Разумов_вер.2.pdf" \
      "2.Как AutoML- и AutoDL-сервисы улучшают реальную разработку_Евгений Смирнов_вер.2.pdf" \
      "3.Обеспечат ли LLM прорыв в эффективности AutoML_Николай Никитин_вер.2.pdf" \
      "4.Где и как использовать LLM в задачах поиска_Валерия Гурьянова_вер.2.pdf" \
      "5.Диффузионные модели для мобильных телефонов_Дмитрий Нестеренко_вер.5.pdf" \
      "6.От промпта к агентной системе_Никита Венедиктов_вер.2.pdf" \
      "7.Валидация в RecSys для корреляции с АВ_Дарья Тихонович_вер.7.pdf" \
      \
      "1.Физически-обоснованное машинное обучение_Александр Хватов_вер.4.pdf" \
      "2.МК по работе с геоданными_Артем Каледин, Денис Афанасьев_вер.5.pdf" \
      "3.ML на графах в e-commerce_Иван Антипов_вер.3.pdf" \
      "4.Синтетика для поиска редких дефектов_Олег Карташев_вер.3.pdf" \
      "5.Как мы развернули трансформер_Артем Карасюк_вер.3.pdf" \
      \
      --provider="openai" --max_concurrent=5

# --- BUSINESS ---
tree -L 3 data/raw/business/
:'
data/raw/business/
├── business_incognita
│   ├── Kept_Подвижной состав РФ_2024 (20 стр).pdf
│   ├── SP_Навигатор_по_мерам_гос_поддержки_2024_74_стр.pdf
│   ├── АИМ_Коммунальное_хозяйство_2024_31_стр.pdf
│   └── РАЭК_Экономика_digital_маркетинга_2024_18_стр.pdf
└── insider_infor
    ├── 3. Тенденции рынка труда 2024.pdf
    ├── AXES_х_Понимаю_Исследование_практик_благополучия_2024.pdf
    ├── dodo-brands-monthly-trading-update-sep-2024.pdf
    └── ЯиП_Энергетический_переход_Вызовы_и_возможности_для_России.pdf
'
# Load all the presentations from POC
poetry run python -m src.chains.run_descriptions process \
"Kept_Подвижной состав РФ_2024 (20 стр).pdf" \
"SP_Навигатор_по_мерам_гос_поддержки_2024_74_стр.pdf" \
"АИМ_Коммунальное_хозяйство_2024_31_стр.pdf" \
"РАЭК_Экономика_digital_маркетинга_2024_18_стр.pdf" \
"3. Тенденции рынка труда 2024.pdf" \
"AXES_х_Понимаю_Исследование_практик_благополучия_2024.pdf" \
"dodo-brands-monthly-trading-update-sep-2024.pdf" \
"ЯиП_Энергетический_переход_Вызовы_и_возможности_для_России.pdf" \
\
--provider="openai" --max_concurrent=5

poetry run python -m src.run_json2chroma convert \
      "Kept_Подвижной состав РФ_2024 (20 стр)" \
      "SP_Навигатор_по_мерам_гос_поддержки_2024_74_стр" \
      "АИМ_Коммунальное_хозяйство_2024_31_стр" \
      "РАЭК_Экономика_digital_маркетинга_2024_18_стр" \
      "3. Тенденции рынка труда 2024" \
      "AXES_х_Понимаю_Исследование_практик_благополучия_2024" \
      "dodo-brands-monthly-trading-update-sep-2024" \
      "ЯиП_Энергетический_переход_Вызовы_и_возможности_для_России" \
      \
      "1.За рамками сценария_Эмели Драль_вер.3 " \
      "2.Kolmogorov Arnold Networks_Павел Плюснин_вер.2 " \
      "2. Пристягина Матрицы компетенций " \
      "3.Система управления процессом окомкования железорудных окатышей_Андрей Голов_вер.2 " \
      "4.Обзор уязвимостей и техник защиты для LLM_Евгений Кокуйкин_вер.3 " \
      "4.Эволюция отбора кандидатов в системе товарных рекомендаций Ozon_Александр Краснов_вер.3 " \
      "6.Увеличиваем число обнаружений в задачах_Дмитрий Колесников_вер.4 " \
      --collection="pres1" --mode="fresh" \
      --provider="openai" --model-name="text-embedding-3-small" \
      --max_concurrent=5


# Embeddings
poetry run python -m src.run_json2chroma convert \
"1.За рамками сценария_Эмели Драль_вер.3" \
"2.Kolmogorov Arnold Networks_Павел Плюснин_вер.2" \
"3.MERA.Text.v.1.2.0. Что под капотом нового релиза_Алена Феногенова_вер.2" \
"4.Обзор уязвимостей и техник защиты для LLM_Евгений Кокуйкин_вер.3" \
"5.Устройство и перспективы использования ML-компиляторов_Виталий Шутов_вер.1" \
"6.Как ML помогает производить лекарства_Владислав Маслов, Василий Вологдин_вер" \
\
"1.Визуальные языковые модели от разбора архитектуры до запуска_Эмиль Шакиров_в" \
"2.Виртуальная примерка одежды для кабинета мерчанта_Роман Лисов_вер.4" \
"3.Система управления процессом окомкования железорудных окатышей_Андрей Голов_" \
"4.Эволюция отбора кандидатов в системе товарных рекомендаций Ozon_Александр Кр" \
"5.Вместо зеленого экрана_Елизавета Петрова_вер.4" \
"6.LLM говорит мультимодальные задачи в речевом домене_Борис Жестков_вер.3" \
"7.Генеративные модели для работы с кодом_Евгений Колесников_вер.1" \
\
"1.Превращаем нейросети в SOTA и для табличных задач_Олег Сидоршин_вер.3" \
"2.Less is more_Дарья Никанорова_вер.3" \
"3.AutoFE-сапер_Валерия Дымбицкая_вер.3" \
"4.Как мы делаем прагматичный поиск и Q&A без LLM_Вадим Захаров_вер.6" \
"5.Фреймворк прикладных инструментов для анализа данных_Ирина Деева_вер.3" \
"6.Увеличиваем число обнаружений в задачах_Дмитрий Колесников_вер.4" \
"7.Мягкая модерация изображений скрыть нельзя блокировать_Юрий Батраков_вер.4" \
\
"1.Разметка датасетов в эпоху мультимодальности_Дмитрий Антипов_вер.4" \
"2. МК «Получение высококачественных данных для обучения ИИ-моделей»_Олег Секач" \
"3.Разметка будущего как GPT помогает обучать модели_Герман Ганус_вер.4" \
"4.Люди не нужны_Данила Бочарников_вер.3" \
"5.Жестовый язык особенности сбора данных, опыт и результаты_Петр Суровцев_вер" \
\
--collection="pres1" --mode="fresh" \
--provider="openai" --model-name="text-embedding-3-small" \
--max_concurrent=5


poetry run python -m src.run_json2chroma convert \ задач_Олег Сидоршин_вер.3" \
      "6.Увеличиваем число обнаружений в задачах_Дмитрий Колесников_вер.4" \
      "4.Обзор уязвимостей и техник защиты для LLM_Евгений Кокуйкин_вер.3" \ \
      "1.За рамками сценария_Эмели Драль_вер.3" \ости_Дмитрий Антипов_вер.4" \
      "4.Эволюция отбора кандидатов в системе товарных рекомендаций Ozon_Александр Краснов_вер.3" \
      "2.Kolmogorov Arnold Networks_Павел Плюснин_вер.2" \_Герман Ганус_вер.4" \
      "РАЭК_Экономика_digital_маркетинга_2024_18_стр" \
      "АИМ_Коммунальное_хозяйство_2024_31_стр" \ опыт и результаты_Петр Суровцев_вер.3" \
      "1.Визуальные языковые модели от разбора архитектуры до запуска_Эмиль Шакиров_вер.3" \
      "6.LLM говорит мультимодальные задачи в речевом домене_Борис Жестков_вер.3" \\
      "7.Генеративные модели для работы с кодом_Евгений Колесников_вер.1" \
      "5.Вместо зеленого экрана_Елизавета Петрова_вер.4" \
      "1.Превращаем нейросети в SOTA и для табличных задач_Олег Сидоршин_вер.3" \
      "3.AutoFE-сапер_Валерия Дымбицкая_вер.3" \
      "4.Как мы делаем прагматичный поиск и Q&A без LLM_Вадим Захаров_вер.6" \
      "1.Разметка датасетов в эпоху мультимодальности_Дмитрий Антипов_вер.4" \
      "2. МК «Получение высококачественных данных для обучения ИИ-моделей»_Олег Секачев_вер.4" \
      "3.Разметка будущего как GPT помогает обучать модели_Герман Ганус_вер.4" \
      "4.Люди не нужны_Данила Бочарников_вер.3" \
      "5.Жестовый язык особенности сбора данных, опыт и результаты_Петр Суровцев_вер.3" \
      "1.Мультимодальные рекомендации в Wildberries_Степан Евстифеев_вер.2" \
      "2.Языковые модели и основы рационального мышления_Ирина Пионтковская_вер.3" \

      --collection="pres1" --mode="fresh" \
      --provider="openai" --model-name="text-embedding-3-small" \
      --max_concurrent=5

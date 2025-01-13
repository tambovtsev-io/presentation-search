import logging
from abc import ABC, abstractmethod
from textwrap import dedent
from typing import Optional, Type, Union

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BasePrompt(ABC):
    """Abstract base class for prompts"""

    def __init__(self):
        """Initialize base prompt"""
        self._prompt_text = self._get_prompt_text()
        self._parser = self._get_parser()
        self._template = self._create_template()

    @abstractmethod
    def _get_prompt_text(self) -> str:
        """Get prompt text. Should include template variables

        Returns:
            String with prompt template
        """
        pass

    @abstractmethod
    def _get_parser(self) -> BaseOutputParser:
        """Get output parser

        Returns:
            Parser for LLM output
        """
        pass

    @abstractmethod
    def _create_template(self) -> ChatPromptTemplate:
        """Create chat template from prompt text

        Returns:
            ChatPromptTemplate ready for chain
        """
        pass

    @property
    def prompt_text(self) -> str:
        """Get prompt text"""
        return self._prompt_text

    @property
    def template(self) -> ChatPromptTemplate:
        """Get chat template"""
        return self._template

    @abstractmethod
    def parse(self, text: str) -> object:
        """Parse LLM output

        Args:
            text: Raw text from LLM

        Returns:
            Parsed output in format specified by concrete prompt
        """
        pass

    def __str__(self) -> str:
        return self._prompt_text


class BaseVisionPrompt(BasePrompt):
    """Base class for vision prompts"""

    def _create_template(self) -> ChatPromptTemplate:
        """Create vision-specific chat template"""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    [
                        {"type": "text", "text": self._prompt_text},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                )
            ]
        )

    def _get_parser(self) -> BaseOutputParser:
        """Get simple string parser by default"""
        return StrOutputParser()

    def parse(self, text: str) -> Union[str, BaseModel, None]:
        """Parse output as simple string by default"""
        return self._parser.parse(text)


class BasePydanticVisionPrompt(BaseVisionPrompt):
    """Base class for vision prompts with Pydantic parsing"""

    def __init__(self):
        """Initialize prompt with Pydantic parser"""
        super().__init__()
        self._format_instructions = self._parser.get_format_instructions()
        self._prompt_with_format = self._prompt_text.format(
            format_instructions=self._format_instructions
        )
        self._template = self._create_template()

    @abstractmethod
    def _get_schema(self) -> Type[BaseModel]:
        """Get Pydantic model for output parsing

        Returns:
            Pydantic model class
        """
        pass

    def _get_parser(self) -> PydanticOutputParser:
        """Get Pydantic parser"""
        return PydanticOutputParser(pydantic_object=self._get_schema())

    def parse(self, text: str) -> Optional[BaseModel]:
        """Parse output according to Pydantic schema"""
        out = self._get_schema()()  # Empty object for schema
        try:
            out = self._parser.parse(text)
        except Exception as e:
            logger.error(f"Error during parsing {text}: {e}")
        return out

    @property
    def prompt_text(self) -> str:
        """Get formatted prompt text with parser instructions"""
        return self._prompt_with_format


########## MY PROMPTS ##########


class SimpleVisionPrompt(BaseVisionPrompt):
    """Simple vision prompt with customizable text"""

    def __init__(self, prompt_text: str = "Describe this slide in detail"):
        """Initialize prompt with custom text

        Args:
            prompt_text: Custom prompt text to use
        """
        self._custom_prompt = prompt_text
        super().__init__()

    def _get_prompt_text(self) -> str:
        """Get prompt text"""
        return self._custom_prompt

    def parse(self, text: str) -> Union[str, BaseModel, None]:
        return None


class JsonH1AndGDPrompt(BasePydanticVisionPrompt):
    """
    Prompt for structured slide analysis with Pydantic parsing.
    H1 And GD means that it parses level 1 headings and
    level 2 of General Description section
    The structure it follows:
    ```
    # Text Content
    [description of text content]

    # Vision Content
    [description of visual content]

    # General Description
    ## Topic Overview

    ## Conclusions and Insights

    ## Layout and Composition
    ```
    """

    class SlideDescription(BaseModel):
        """
        Slide Description Schema

        The properties are set to empty strings by default
        to handle failed parsing
        """

        class GeneralDescription(BaseModel):
            topic_overview: str = ""
            conclusions_and_insights: str = ""
            layout_and_composition: str = ""

        text_content: str = ""
        visual_content: str = ""
        general_description: GeneralDescription = GeneralDescription()

    def _get_schema(self) -> Type[BaseModel]:
        """Get output schema"""
        return self.SlideDescription

    def _get_prompt_text(self) -> str:
        """Get prompt text"""
        return dedent(
            """
You are a presentation slide description agent. Your task is to provide a detailed description of the slide in a structured format. Analyze and describe both visual and textual elements according to the following structure. Provide the description in Russian language.

Adapt the level of detail in your description based on slide complexity:
- For simple slides (1-2 elements): Focus on main message and basic layout
- For medium complexity slides (3-5 elements): Include basic styling and relationships between elements
- For complex slides (6+ elements): Provide full detailed description including all styling, accessibility features, and relationships

# Text Content
## Text Elements
Describe all text elements present, including but not limited to:
- Headers/titles
- Body text
- Lists
- Captions
- Labels

## Text Styling
Describe text styling only when it serves a specific purpose:
- Bold/italic text when used for emphasis
- Color coding when indicating categories or relationships
- Font size variations when establishing information hierarchy
- Background highlighting when drawing attention to key points
Note: Explain how each styling choice contributes to understanding the content

## Multilingual Elements
### Language Guidelines
When describing slides containing multiple languages:
- Identify the primary language (Russian) and secondary language elements
- Specify the languages used
- Format bilingual elements as:
    * Russian term / English term
    * Add explanatory notes for context when necessary

### Specific Terms
- For technical terms:
    * Provide both Russian translation and original English term in parentheses
    * Example: "Облачное хранилище (Cloud Storage)"
- For industry-specific terminology:
    * Keep widely accepted English terms unchanged
    * Example: "AI", "API", "CEO", "ML"
- For names of companies:
    * Яндекс (Yandex)
    * Озон (OZON)
    * Вайлдберрис (Wildberries)
    * Ламода (Lamoda)
    * Авито (Avito)
    * ВКонтакте (VKontakte)
    * Сбер (Sber)
    * МТС (MTS)
    * РЖД (RZD)
- For charts and graphs:
    * Note language of axes labels, legends, and data points
    * Specify if measurements use local or international notation
- For Data Science and AI terminology:
    * Machine Learning (ML) terms:
    - "Машинное обучение (Machine Learning, ML)"
    - "Обучение с учителем (Supervised Learning)"
    - "Нейронные сети (Neural Networks)"
    - Common metrics: "Точность (Accuracy)", "Полнота (Recall)", "F1-мера (F1-score)"
    * Computer Vision (CV) terms:
    - "Компьютерное зрение (Computer Vision, CV)"
    - "Распознавание объектов (Object Detection)"
    - "Сегментация изображений (Image Segmentation)"
    - "Классификация изображений (Image Classification)"
    * Natural Language Processing (NLP) terms:
    - "Обработка естественного языка (Natural Language Processing, NLP)"
    - "Векторное представление слов (Word Embeddings)"
    - "Анализ тональности (Sentiment Analysis)"
    - "Языковые модели (Language Models)"

# Visual Content
Provide detailed descriptions of all visual components present on the slide. Do not leave any details un-narrated as some of your viewers are vision-impaired, so if you don't narrate every number they won't know the number.

When multiple visual elements are present, describe them in the following order:
1. Primary functional visuals (main charts, diagrams, or images that convey key information)
2. Secondary functional visuals (supporting graphics or illustrations)
3. Decorative elements (background patterns, borders, or design elements)

For each visual element, describe:
- What it represents or shows
- Its location on the slide
- Its relationship to other elements
- Key details that contribute to the slide's message

## Charts and Graphs
For charts and graphs, additionally describe:
- All data labels and their positioning
- Annotations and callouts
- Scale markers and units
- Any highlighting or special marking of specific data points
- Legend entries and their relationship to the data

## Tables:
For tables, provide detailed description of it's content:
- Table structure (rows, columns, merged cells)
- Identify the relationship represented by the table
- Analyze key data points and metrics
- Provide key insights from the data
- Note any color coding or special formatting

Remember to focus on the story the table tells rather than listing every value.

## Accessibility and Visualisation
For accessibility and visualization features note presence or absence of:
- Patterns or textures alongside colors
- Alternative visual indicators
- High-contrast elements
- Any other features aiding in data interpretation

## Branding
For branded elements (if present):
- Corporate colors and their usage
- Brand-specific visual elements
- Consistent design patterns
- Typography aligned with brand guidelines
- Position and prominence of brand elements

For minimally branded slides:
- Focus on functional elements
- Note basic styling choices
- Describe any subtle branding elements present

## Visual Elements Interaction
When describing relationships between visual elements, include:
- Hierarchy and dependencies:
    * Primary-secondary relationships
    * Supporting elements
    * Connected or linked elements
- Visual flow:
    * Direction of information flow (left-to-right, top-to-bottom, circular)
    * Sequential relationships
    * Cause-and-effect relationships
- Interactive elements:
    * Elements that reference each other
    * Cross-referencing between charts/text
    * Complementary information presentation

# General Description
## Topic Overview
- Topic
- Purpose
- Key Information

## Conclusions and Insights
### Key Takeaways
- Main message or story the slide is conveying
- Critical patterns or trends in the data
- Significant correlations or relationships

### Data Analysis
- Notable changes or anomalies
- Benchmark comparisons (if present)
- Potential implications of the presented data

## Layout and Composition
Describe the overall arrangement of elements on the slide, including:
- Spatial organization (left/right, top/bottom)
- Proportions and emphasis
- Use of white space

# Output Format
Provide your description as a JSON object with the following high-level structure:

{{
  "text_content": "string",     // Content from "Text Content" section
  "visual_content": "string",   // Content from "Visual Content" section
  "general_description": {{     // Content from "General Description" section
    "topic_overview": "string",
    "conclusions_and_insights": "string",
    "layout_and_composition": "string"
  }}
}}

All text should be in Russian language. Preserve paragraph breaks using \n characters.

Example of a Simple Slide:
```json
{{
  "text_content": "Заголовок: \"Рост продаж в Q3 2023\"\nОсновная цифра: \"25%\"\nПодпись: \"Квартальный рост\"\n\nСтилизация текста: крупный размер шрифта использован для акцента на цифре \"25%\"",

  "visual_content": "Минимальное оформление без дополнительных визуальных элементов.",

  "general_description": {{
    "topic_overview": "Тема: Квартальные показатели продаж\nЦель: Представить ключевые цифры за квартал\nКлючевая информация: Рост продаж на 25%",
    "conclusions_and_insights": "",
    "layout_and_composition": "Центрированная композиция с одним крупным числом в середине слайда"
  }}
}}
```

Example of a Medium-Complexity Slide:
```json
{{
  "text_content": "Заголовок: \"Структура операционных расходов за 2023 год\"\n\nЭлементы легенды:\n- Зарплата и бонусы (45%)\n- Аренда помещений (20%)\n- IT-инфраструктура (15%)\n- Маркетинг (12%)\n- Прочие расходы (8%)\n\nСноска: \"По данным финансового отдела, Q4 2023\"\n\nСтилизация текста: заголовок синий цвет, 28pt\n\nМногоязычные элементы:\n- Операционные расходы (Operational Expenses, OPEX)\n- IT-инфраструктура (IT Infrastructure)\n- Cloud-сервисы (Cloud Services)\nФинансовые показатели используют международный формат",

  "visual_content": "Круговая диаграмма с 5 секторами различных цветов. Каждый сектор имеет подпись с процентным значением. Наибольший сектор (зарплата и бонусы) слегка выдвинут для акцента.\n\nЦветовая схема:\n- Синий - зарплата и бонусы\n- Зеленый - аренда\n- Оранжевый - IT\n- Желтый - маркетинг\n- Серый - прочие расходы\n\nТонкая разделительная линия между заголовком и основным содержанием. Небольшой логотип компании в правом нижнем углу",

  "general_description": {{
    "topic_overview": "Тема: Структура расходов компании\nЦель: Визуализировать основные категории расходов\nКлючевая информация: Распределение бюджета по 5 ключевым категориям",
    "conclusions_and_insights": "Расходы на персонал составляют почти половину бюджета (45%)\nIT-инфраструктура и маркетинг совместно потребляют более четверти бюджета (27%)\nДоля прочих расходов относительно невелика (8%)\n\nСтруктура расходов типична для компаний сферы услуг\nЗначительная доля IT-расходов указывает на высокий уровень цифровизации\nСоотношение постоянных и переменных затрат оптимально",
    "layout_and_composition": "Круговая диаграмма занимает центральную часть (70% площади)\nЗаголовок вверху\nЛегенда справа от диаграммы\nСноска с источником данных внизу"
  }}
}}
```

Example of a Complex Slide:
```json
{{
  "text_content": "Основной заголовок: \"Потрясения последних лет уже кардинально изменили мир и повлияют на энергетические рынки будущего\"\n\nПодзаголовок: \"Зависимость цепочек поставок и инвестиций в энергетике от геополитических событий\"\n\nТекстовые блоки:\n1. \"Мир, каким мы его знали\":\n- Экономия масштаба в производстве и ресурсоснабжении\n- Эффективные глобальные цепочки поставок\n- Потребление «золотым миллиардом»\n- Акционерная стоимость превыше всего\n\n2. \"Мир, к которому мы идем\":\n- Разрывы привычных цепочек поставок, многополярность экономического ландшафта\n- Ориентация на снижение рисков через построение суверенных энергетических экосистем\n- Расширение практики приоритизации дружественных отношений в энергетических сделках\n- Акцент на доступную и устойчивую энергию\n\nИсточник: \"Federal Reserve Bank of New York; Bloomberg; открытые источники\"",

  "visual_content": "Линейный график с двумя осями Y:\n- Черная линия: Индекс давления на глобальные цепочки поставок\n- Желтая линия: Глобальный индекс цен на энергию\n\nОсь X: Временная шкала 2000-2023\n\nАннотации важных событий:\n- Мировой экономический кризис\n- «Арабская весна»\n- BREXIT\n- Начало торговой войны США и Китая\n- Эпидемия COVID-19\n- Начало СВО\n- Палестинский конфликт\n\nИспользуются различные паттерны линий для доступности\nЛоготип \"Яков и Партнёры\" в правом нижнем углу\nФирменная цветовая схема с синим и желтым цветами",

  "general_description": {{
    "topic_overview": "Тема: Влияние геополитических событий на энергетические рынки\nЦель: Показать взаимосвязь между глобальными событиями, цепочками поставок и ценами на энергоносители\nКлючевая информация: Сравнение прошлого и будущего состояния энергетических рынков",
    "conclusions_and_insights": "Потрясения последних лет значительно изменили структуру энергетических рынков\nВажность адаптации к новым условиям и рискам в цепочках поставок\nНеобходимость перехода к устойчивым и доступным источникам энергии\n\nНаблюдаются значительные изменения в индексах\nСравнение с предыдущими периодами показывает тренды\nВыявлены потенциальные последствия для бизнеса",
    "layout_and_composition": "Левая часть (60%): график и заголовок\nПравая часть (40%): текстовые блоки на черном фоне\nПрисутствуют фирменные элементы оформления"
  }}
}}
```

Example of a Slide With Table:
```json
{{
  "text_content": "Заголовок: \"Промышленность\"\nПодзаголовок: \"Проектное финансирование промышленного производства (2/3)\"\n\nПояснительный текст:\nНа данный момент ФРП реализует 7 программ, направленных на поддержку развития компонентной базы, цифровизации предприятий, повышения производительности труда и др.\n\nСтилизация текста:\n- Заголовки выделены жирным шрифтом\n- Числовые значения представлены в деловом формате\n- Используется иерархическая система шрифтов для различных уровней информации",

  "visual_content": "Таблица программ финансирования ФРП, состоящая из 7 колонок и 5 программ.\n\nКлючевые характеристики:\n- Охватывает основные параметры: суммы займов, ставки, сроки и условия\n- Особый акцент на программу автокомпонентов с максимальной суммой займа до 5 млрд руб.\n- Базовые ставки варьируются от 3% до 5%\n\nВажные закономерности:\n- Более крупные займы имеют более длительные сроки погашения\n- Программы с госгарантией предлагают сниженные ставки\n- Требования к софинансированию увеличиваются с размером займа\n\nВыделяющиеся условия:\n- Программа автокомпонентов предлагает максимальный срок в 7 лет\n- Минимальный порог входа от 50 млн руб. для базовых программ"

  "general_description": {{
    "topic_overview": "Тема: Проектное финансирование промышленного производства\nЦель: Представить детальную информацию о программах кредитования ФРП\nКлючевая информация: Условия и параметры 5 основных программ кредитования для промышленных предприятий",
    "conclusions_and_insights": "Основные выводы:\n- ФРП предлагает разнообразные программы с льготными условиями кредитования\n- Процентные ставки варьируются от 3% до 5%\n- Сроки кредитования достигают 7 лет\n- Суммы займов находятся в широком диапазоне от 5 млн до 5 млрд рублей\n- Каждая программа имеет специфические требования по софинансированию и целевым показателям\n\nКлючевые особенности программ:\n- Возможность получения второго займа для масштабирования\n- Гибкие условия по процентным ставкам при банковской гарантии\n- Особый фокус на производство автокомпонентов с максимальными суммами займа\n- Поддержка лизинговых операций с особыми условиями финансирования",
    "layout_and_composition": "Слайд имеет четкую иерархическую структуру:\n- Заголовок и подзаголовок в верхней части\n- Основная таблица занимает центральную часть слайда\n- Брендинговые элементы расположены по периметру\n- Использовано эффективное распределение пространства для большого объема данных"
  }}
}}
```
"""
        )


class JsonH1AndGDPromptEng(BasePydanticVisionPrompt):
    """The copy of the above but in English"""

    class SlideDescription(BaseModel):
        class GeneralDescription(BaseModel):
            topic_overview: str
            conclusions_and_insights: str
            layout_and_composition: str

        text_content: str
        visual_content: str
        general_description: GeneralDescription

    def _get_schema(self) -> Type[BaseModel]:
        """Get output schema"""
        return self.SlideDescription

    def _get_prompt_text(self) -> str:
        """Get prompt text"""
        return dedent(
            """
You are a presentation slide description agent. Your task is to provide a detailed description of the slide in a structured format. Analyze and describe both visual and textual elements according to the following structure. The description should be in English, while preserving Russian text elements from the slide with translations where appropriate.

Adapt the level of detail in your description based on slide complexity:
- For simple slides (1-2 elements): Focus on main message and basic layout
- For medium complexity slides (3-5 elements): Include basic styling and relationships between elements
- For complex slides (6+ elements): Provide full detailed description including all styling, accessibility features, and relationships

# Text Content
## Text Elements
Describe all text elements present, including but not limited to:
- Headers/titles (provide both Russian original and English translation)
- Body text (provide both Russian original and English translation)
- Lists
- Captions
- Labels
Format: "Russian text" (English translation)

## Text Styling
Describe text styling only when it serves a specific purpose:
- Bold/italic text when used for emphasis
- Color coding when indicating categories or relationships
- Font size variations when establishing information hierarchy
- Background highlighting when drawing attention to key points
Note: Explain how each styling choice contributes to understanding the content

## Multilingual Elements
### Language Guidelines
When describing slides containing multiple languages:
- Identify the primary language (Russian) and secondary language elements
- Specify the languages used
- Format bilingual elements as:
    * Russian term / English term
    * Add explanatory notes for context when necessary

### Specific Terms
- For technical terms:
    * Provide English term with Russian translation in parentheses
    * Example: "Cloud Storage (Облачное хранилище)"
- For industry-specific terminology:
    * Keep widely accepted English terms unchanged
    * Example: "AI", "API", "CEO", "ML"
- For charts and graphs:
    * Note language of axes labels, legends, and data points
    * Provide translations for Russian labels
- For names of companies:

- For Data Science and AI terminology:
    * Machine Learning (ML) terms:
    - "Machine Learning / Машинное обучение"
    - "Supervised Learning / Обучение с учителем"
    - "Neural Networks / Нейронные сети"
    - Common metrics: "Accuracy / Точность", "Recall / Полнота", "F1-score / F1-мера"
    * Computer Vision (CV) terms:
    - "Computer Vision / Компьютерное зрение"
    - "Object Detection / Распознавание объектов"
    - "Image Segmentation / Сегментация изображений"
    - "Image Classification / Классификация изображений"
    * Natural Language Processing (NLP) terms:
    - "Natural Language Processing / Обработка естественного языка"
    - "Word Embeddings / Векторное представление слов"
    - "Sentiment Analysis / Анализ тональности"
    - "Language Models / Языковые модели"

# Visual Content
Provide detailed descriptions of all visual components present on the slide. When multiple visual elements are present, describe them in the following order:
1. Primary functional visuals (main charts, diagrams, or images that convey key information)
2. Secondary functional visuals (supporting graphics or illustrations)
3. Decorative elements (background patterns, borders, or design elements)

For each visual element, describe:
- What it represents or shows
- Its location on the slide
- Its relationship to other elements
- Key details that contribute to the slide's message

When describing visual elements containing Russian text, provide translations:
- Chart labels: "Russian label" (English translation)
- Annotations: preserve original with translation
- Data labels: include both languages if present

## Charts and Graphs
For charts and graphs, additionally describe:
- All data labels and their positioning
- Annotations and callouts
- Scale markers and units
- Any highlighting or special marking of specific data points
- Legend entries and their relationship to the data

## Tables:
For tables, provide detailed description of it's content:
- Table structure (rows, columns, merged cells)
- Identify the relationship represented by the table
- Analyze key data points and metrics
- Provide key insights from the data
- Note any color coding or special formatting

## Accessibility and Visualisation
For accessibility and visualization features note presence or absence of:
- Patterns or textures alongside colors
- Alternative visual indicators
- High-contrast elements
- Any other features aiding in data interpretation

## Branding
For branded elements (if present):
- Corporate colors and their usage
- Brand-specific visual elements
- Consistent design patterns
- Typography aligned with brand guidelines
- Position and prominence of brand elements

For minimally branded slides:
- Focus on functional elements
- Note basic styling choices
- Describe any subtle branding elements present

## Visual Elements Interaction
When describing relationships between visual elements, include:
- Hierarchy and dependencies:
    * Primary-secondary relationships
    * Supporting elements
    * Connected or linked elements
- Visual flow:
    * Direction of information flow (left-to-right, top-to-bottom, circular)
    * Sequential relationships
    * Cause-and-effect relationships
- Interactive elements:
    * Elements that reference each other
    * Cross-referencing between charts/text
    * Complementary information presentation

# General Description
## Topic Overview
- Topic
- Purpose
- Key Information

## Conclusions and Insights
### Key Takeaways
- Main message or story the slide is conveying
- Critical patterns or trends in the data
- Significant correlations or relationships

### Data Analysis
- Notable changes or anomalies
- Benchmark comparisons (if present)
- Potential implications of the presented data

## Layout and Composition
Describe the overall arrangement of elements on the slide, including:
- Spatial organization (left/right, top/bottom)
- Proportions and emphasis
- Use of white space

# Output Format
Provide your description as a JSON object with the following high-level structure:

{{
  "text_content": "string",   // Content from "Text Content" section, with Russian text preserved and translated
  "visual_content": "string   // Content from "Visual Content" section in English, with Russian elements translated
  "general_description": {{   // Content from "General Description" section in English
    "topic_overview": "string",
    "conclusions_and_insights": "string",
    "layout_and_composition": "string"
  }}
}}

All descriptive text should be in English. When including Russian text elements from the slide, preserve them in their original form with English translations in parentheses. Preserve paragraph breaks using \n characters.

Example of a Simple Slide:
```json
{{
  "text_content": "Title (in Russian): \"Рост продаж в Q3 2023\" (Sales Growth in Q3 2023)\nMain figure: \"25%\"\nCaption (in Russian): \"Квартальный рост\" (Quarterly Growth)\n\nText styling: Large font size used to emphasize the \"25%\" figure",

  "visual_content": "Minimal design without additional visual elements. The absence of supporting visual elements doesn't impede information perception.",

  "general_description": {{
  "topic_overview": "Topic: Quarterly sales metrics\nPurpose: Present key quarterly figures\nKey Information: 25% sales growth",
    "conclusions_and_insights": "",
    "layout_and_composition": "Centered composition with one large number in the middle of the slide"
  }}
}}
```

Medium-complexity example:
```json
{{
  "text_content": "Title (in Russian): \"Структура операционных расходов за 2023 год\" (Operational Expenses Structure for 2023)\n\nLegend elements (in Russian):\n- Зарплата и бонусы (Salaries and Bonuses) (45%)\n- Аренда помещений (Facility Rent) (20%)\n- IT-инфраструктура (IT Infrastructure) (15%)\n- Маркетинг (Marketing) (12%)\n- Прочие расходы (Other Expenses) (8%)\n\nFootnote (in Russian): \"По данным финансового отдела, Q4 2023\" (According to Financial Department, Q4 2023)\n\nText styling: Blue title, 28pt\n\nBilingual elements:\n- Operational Expenses (OPEX) / Операционные расходы\n- IT Infrastructure / IT-инфраструктура\n- Cloud Services / Cloud-сервисы\nFinancial indicators use international format",

  "visual_content": "Pie chart with 5 differently colored sectors. Each sector includes a percentage label. The largest sector (salaries and bonuses) is slightly offset for emphasis.\n\nColor scheme:\n- Blue - salaries and bonuses\n- Green - rent\n- Orange - IT\n- Yellow - marketing\n- Gray - other expenses\n\nThin dividing line between header and main content. Small company logo in bottom right corner",

  "general_description": {{
    "topic_overview": "Topic: Company expense structure\nPurpose: Visualize main expense categories\nKey Information: Budget distribution across 5 key categories",
    "conclusions_and_insights": "Personnel expenses constitute nearly half the budget (45%)\nIT infrastructure and marketing jointly consume over a quarter of the budget (27%)\nOther expenses portion is relatively small (8%)\n\nExpense structure is typical for service companies\nSignificant IT expenses indicate high digitalization level\nFixed to variable costs ratio is optimal",
    "layout_and_composition": "Pie chart occupies central area (70% of space)\nHeader at top\nLegend to the right of chart\nData source footnote at bottom"
  }}
}}
```

Complex example:
```json
{{
    "text_content": "Main heading (in Russian): \"Потрясения последних лет уже кардинально изменили мир и повлияют на энергетические рынки будущего\" (Recent years' upheavals have radically changed the world and will affect future energy markets)\n\nSubheading (in Russian): \"Зависимость цепочек поставок и инвестиций в энергетике от геополитических событий\" (Dependency of supply chains and energy investments on geopolitical events)\n\nText blocks (in Russian):\n1. \"Мир, каким мы его знали\" (The world as we knew it):\n[Russian text preserved with key points translated]\n\n2. \"Мир, к которому мы идем\" (The world we're heading to):\n[Russian text preserved with key points translated]\n\nSource (in Russian): \"Federal Reserve Bank of New York; Bloomberg; открытые источники\"",

    "visual_content": "Linear graph with two Y-axes:\n- Black line: Global Supply Chain Pressure Index\n- Yellow line: Global Energy Price Index\n\nX-axis: Timeline 2000-2023\n\nEvent annotations:\nGlobal Economic Crisis\nArab Spring\nBREXIT\nUS-China Trade War start\nCOVID-19 epidemic\nSpecial Military Operation\nPalestinian conflict\n\nDifferent line patterns used for accessibility\n\"Yakov and Partners\" logo in bottom right\nCorporate color scheme with blue and yellow",

    "general_description": {{
    "topic_overview": "Topic: Geopolitical events' impact on energy markets\nPurpose: Demonstrate relationship between global events, supply chains, and energy prices\nKey Information: Comparison of past and future states of energy markets",
    "conclusions_and_insights": "Recent upheavals have significantly changed energy market structure\nImportance of adapting to new conditions and supply chain risks\nNecessity of transitioning to sustainable and accessible energy sources\n\nSignificant changes observed in indices\nComparison with previous periods shows trends\nPotential business implications identified",
    "layout_and_composition": "Left side (60%): graph and header\nRight side (40%): text blocks on black background\nCorporate design elements present"
    }}
}}
```

Example of a Slide With Table:
```json
{{
  "text_content": "Title (in Russian): \"Промышленность\" (Industry)\nSubtitle (in Russian): \"Проектное финансирование промышленного производства (2/3)\" (Project Financing for Industrial Production (2/3))\n\nExplanatory text (in Russian):\n\"На данный момент ФРП реализует 7 программ, направленных на поддержку развития компонентной базы, цифровизации предприятий, повышения производительности труда и др.\" \n(Currently, IDF implements 7 programs aimed at supporting component base development, enterprise digitalization, labor productivity improvement, etc.)\n\nText styling:\n- Headers in bold\n- Numerical values in business format\n- Hierarchical font system for different information levels",

  "visual_content": "Main visual element is a large information table occupying the central part of the slide.\n\nTable structure:\nColumns:\n1. Main lending programs\n2. Loan amount, mln RUB\n3. Interest rate\n4. Max loan term\n5. Total project budget, mln RUB\n6. Co-financing\n7. Special conditions\n\nRow content (preserving Russian original with translations):\n1. Components Manufacturing (1st loan):\n- Loan amount: 100–1,000 mln RUB\n- Rate: 3% (bank guarantee) / 5% base\n- Term: 5 years\n- Budget: from 125 mln RUB\n- Special conditions: targeted sales ≥ 30% of loan\n\n2. Components Manufacturing (2nd loan):\n- Amount: 50–500 mln RUB\n- Rate: 5%\n- Term: 3 years\n- Special conditions: up to 50% of first loan\n\n3. Labor Productivity:\n- Amount: 50–300 mln RUB\n- Rate: 3% (bank guarantee) / 5% base\n- Term: 5 years\n- Budget: from 62.5 mln RUB\n- Co-financing: ≥ 20%\n- Conditions: productivity growth 10-30%\n\n4. Automotive Components:\n- Amount: 100–5,000 mln RUB\n- Rate: 3% (bank guarantee) / 5% base\n- Term: 7 years\n- For units and assemblies production\n\n5. Leasing:\n- Amount: 5–500 mln RUB\n- Rate: 5%\n- Term: 5 years\n- Co-financing: ≥ 55%\n- Conditions: 10-90% of leasing advance\n\nTable styling:\n- Alternating row background colors\n- Right-aligned numerical data\n- Left-aligned text information\n\nBranding:\n- IDF logo in top right corner\n- Footer text: \"Leading Russian Strategic Consultant\"\n- Logo and name \"Strategy Partners | strategy.ru\"\n- Slide number: 9",

  "general_description": {{
    "topic_overview": "Topic: Project financing for industrial production\nPurpose: Present detailed information about IDF lending programs\nKey Information: Terms and parameters of 5 main lending programs for industrial enterprises",

    "conclusions_and_insights": "Key findings:\n- IDF offers diverse programs with preferential lending terms\n- Interest rates range from 3% to 5%\n- Loan terms extend up to 7 years\n- Loan amounts range from 5 mln to 5 bln rubles\n- Each program has specific co-financing requirements and target indicators\n\nKey program features:\n- Possibility of second loan for scaling\n- Flexible interest rates with bank guarantee\n- Special focus on automotive components with maximum loan amounts\n- Support for leasing operations with special financing conditions",

    "layout_and_composition": "Slide has clear hierarchical structure:\n- Header and subheader at top\n- Main table occupies central area\n- Branding elements positioned around perimeter\n- Efficient space distribution for large data volume"
  }}
}}
```
"""
        )

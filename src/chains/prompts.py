from typing import Dict, Any, Optional, Type, Union, TypeVar
from abc import ABC, abstractmethod
import logging

from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers import StrOutputParser

from textwrap import dedent

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
    def parse(self, text: str) -> T:
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
        return ChatPromptTemplate.from_messages([
            ("human", [
                {"type": "text", "text": self._prompt_text},
                {
                    "type": "image",
                    "image_url": "data:image/png;base64,{image_base64}"
                }
            ])
        ])

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
        out = None
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
    """Prompt for structured slide analysis with Pydantic parsing"""
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
    * Example: "AI", "API", "CEO"
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
Provide detailed descriptions of all visual components present on the slide. When multiple visual elements are present, describe them in the following order:
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

  "visual_content": "Минимальное оформление без дополнительных визуальных элементов. Отсутствие вспомогательных визуальных элементов не мешает восприятию информации.",

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
""")

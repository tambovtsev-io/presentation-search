# Presentation-RAG

## Ссылки
### Навигация по проекту
- [System Design Document](/docs/system_design_doc.md)
- [Документация](/docs)
- [Ноутбуки](/notebooks/)

### Raftds
- [Jira](https://jira.raftds.com/secure/RapidBoard.jspa?rapidView=98&projectKey=RMI&selectedIssue=RMI-40)
- [Confluence](https://confluence.raftds.com/display/RMI1/Raft+ML+internship+1)
- [GitLab · Presentation RAG](https://gitlab.raftds.com/ilia.tambovtsev/presentation-rag)

## Технические моменты
### Работа с ветками
Ветки:
- [main](https://gitlab.raftds.com/ilia.tambovtsev/presentation-rag/-/tree/main?ref_type=heads)
- [POC](https://gitlab.raftds.com/ilia.tambovtsev/presentation-rag/-/tree/POC?ref_type=heads) - Proof of Concept

В проекте два типа веток. 

`RMI-XX-<task>` отражают задачи из доски Jira. 

Ветка `POC` - рабочая ветка на текущий момент. В `POC` маленький датасет + есть задачи по ресерчу. Возможно не все из `POC` пойдет в `main`. 

### Работа с данными
В проекте используется dvc. Для каждой задачи подгружается свой датасет. Идея: не надо загружать 200 презентаций, если хочется перепроверить тестовую задачу с 10 слайдами. [Подробнее про dvc](./docs/workflow/data_version_control.md)

## Progress
[Гугл-диск с презентациями](https://drive.google.com/drive/folders/1IvUsxxtyyTuHdZff9szhd2OtIATRTcG4?usp=sharing)

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

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
В проекте используется dvc. Для каждой задачи подгружается свой датасет. Идея: не надо загружать 200 презентаций, если хочется перепроверить тестовую задачу с 10 слайдами. [Подробнее про dvc](./docs/workflow/dvc)
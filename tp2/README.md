# Trabalho Prático 2 - Retail Vision Intelligence System

### Trabalho realizado por: ###
* Leonor Rebola (leonor.rebola@ubi.pt)
* Número: 53663
* Curso: Inteligência Artificial e Ciência de Dados
* UC: Interação com Modelos em Larga Escala
---

## Descrição do trabalho
Este projeto implementa um sistema de inspeção contínua de prateleiras de supermercado com memória histórica, capaz de analisar imagens com um modelo de linguagem multimodal, aprender as regras do gestor em linguagem natural e integrar a análise visual com dados de trajetória do Projeto 1.

---

## Estrutura do trabalho
```
tp2/
├── data/
│   ├── annotations/        — ground truth das imagens anotadas
│   ├── images/             — dataset de imagens (SKU-110K + fotos próprias)
│   ├── inspections/        — inspection records gerados
│   ├── inspections_rag/    — inspeções selecionadas para indexação RAG
│   ├── reports/            — relatórios gerados automaticamente
│   ├── rules/              — regras persistidas em disco
│   └── journeys.csv        — dados de trajetória do Projeto 1
├── src/
│   ├── shelf_inspector.py
│   ├── rule_engine.py
│   ├── rag_memory.py
│   ├── report_generator.py
│   ├── interface.py
│   └── journey_context.py
├── prompts/
│   ├── prompt_A_zero_shot.txt
│   ├── prompt_B_cot.txt
│   ├── prompt_C_few_shot.txt
│   ├── prompt_rule_conversion.txt
│   ├── prompt_rag_summary.txt
│   ├── prompt_rag_answer.txt
│   ├── prompt_report.txt
│   └── prompt_llm_judge.txt
├── vectorstore/            — índice FAISS persistente
├── cache/                  — cache de resultados da API
├── evaluate.py
├── evaluation_report.json
├── .env.example
├── README.md
└── requirements.txt
```
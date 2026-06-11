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

---

## Modelos utilizados
- **LLM**: `gemini-2.5-flash` via Google AI Studio API
- **Embeddings**: `gemini-embedding-001` (dimensão 3072)
- **Temperature**: 0 (resultados reprodutíveis)
- **Estratégias de prompting**: Zero-shot (A), Chain-of-Thought (B), Few-shot (C)

---

## Requisitos

1. Instalar os requirements
```bash
pip install -r requirements.txt
```

2. Configurar a chave de API
```bash
cp .env.example .env
# editar .env e adicionar a chave GEMINI_API_KEY
```

**Nota:** É necessária uma chave de API do Google AI Studio (gratuita). 
Criar conta em https://aistudio.google.com e obter a chave em "Get API Key".

---

## Como Executar

### Interface conversacional (modo recomendado)
```bash
python src/interface.py
```

**Comandos disponíveis na interface:**
```
inspect <imagem.jpg> [zona] [estrategia]   — inspeciona uma imagem
add rule "<regra em português>"            — adiciona uma regra
list rules                                 — lista todas as regras
history "<pergunta>"                       — consulta histórico RAG
report --dir <pasta>                       — gera relatório
stats                                      — estatísticas do índice
help                                       — mostra todos os comandos
```

### Componentes individuais

Inspecionar uma imagem:
```bash
python src/shelf_inspector.py data/images/IMG_9058.jpg Z_S6 A
```

Correr as 3 estratégias na mesma imagem:
```bash
python src/shelf_inspector.py compare data/images/IMG_9058.jpg Z_S6
```

Avaliar todas as imagens anotadas:
```bash
python src/shelf_inspector.py eval A
python src/shelf_inspector.py eval B
python src/shelf_inspector.py eval C
```

Adicionar uma regra:
```bash
python src/rule_engine.py add "Na zona Z_S1, se o fill rate cair abaixo de 60%, nível crítico"
```

Indexar inspeções no RAG:
```bash
python src/rag_memory.py index-dir data/inspections_rag
```

Consultar histórico:
```bash
python src/rag_memory.py query "quando foi a última vez que Z_S1 teve problemas?"
```

Gerar relatório:
```bash
python src/report_generator.py --split data/inspections
```

### Harness de avaliação
```bash
python evaluate.py --images-dir data/images --output evaluation_report.json
```

Para avaliar com dataset externo:
```bash
python evaluate.py --images-dir test_images/ --output evaluation_report.json --ground-truth test_ground_truth.json
```

---

## Outputs
1. **data/inspections/**: inspection records gerados pelo shelf_inspector
2. **data/rules/**: regras convertidas e persistidas em disco
3. **vectorstore/**: índice FAISS com 77 inspeções indexadas
4. **data/reports/**: relatórios de inspeção em Markdown
5. **evaluation_report.json**: métricas de avaliação do sistema
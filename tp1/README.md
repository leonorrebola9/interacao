# Trabalho prático 1 - From Raw Detections to Real Intelligence

### Trabalho realizado por: ###
* Leonor Rebola (leonor.rebola@ubi.pt)
* Número: 53663

## Descrição do trabalho
Este projeto implementa um pipeline de análise de dados de uma loja de retalho que transforma eventos brutos de deteção por câmara em inteligência operacional útil para um gestor de loja.
O sistema recebe um stream de eventos anónimos (sem identidade de pessoa) e produz automaticamente um briefing semanal com insights acionáveis, anomalias detetadas e recomendações concretas.

## Estrutura do trabalho
O trabalho segue a seguinte estrutura:

```
tp1/
├── data/
│   ├── events.csv
│   └── zones.json
├── src/
│   ├── stitcher.py
│   ├── analytics.py
│   ├── insights.py
│   └── report.py
├── output/
│   ├── journeys.csv
│   ├── metrics.json
│   ├── insights.json
│   └── weekly_report.md
├── prompts/
│   ├── strategy_A_zero_shot.txt
│   ├── strategy_B_few_shot.txt
│   └── report_prompt.txt
├── evaluate.py
├── README.md
└── requirements.txt
```

## Modelos utilizados
- **LLM**: `llama3.1:8b` via Ollama
- **Temperature**: 0.0 (resultados reprodutíveis)
- **Estratégias de prompting**: Zero-shot (A) e Few-shot (B) (ver pasta `prompts/`)

## Módulos
O trabalho está dividido em 4 módulos sequenciais:
1. **stitcher.py**: reconstrói trajetórias individuais a partir de eventos anónimos
2. **analytics.py**: calcula métricas de tráfego, zonas, funil e anomalias
3. **insights.py**: gera insights acionáveis em JSON via LLM local (Ollama)
4. **report.py**: gera o briefing semanal em formato .md

## Requisitos
Para que o trabalho corra da forma correta, tem de se seguir os seguintes passos:

1. Instalar os requirements
```bash
pip install -r requirements.txt
```

2. Instalar o LLM
```bash
ollama pull llama3.1:8b
```
Para correr: 'ollama serve'

**Nota:** Para poder ser usado corretamente, o modelo Ollama tem de estar instalado e a correr antes de executar os ficheiros 'insights.py' e 'report.py'

## Como Executar
1. Pipeline completo (passo a passo)
```bash
python src/stitcher.py --input data/events.csv --output output/journeys.csv --zones data/zones.json
 
python src/analytics.py --input output/journeys.csv --output output/metrics.json
 
python src/insights.py --input output/metrics.json --output output/insights.json
 
python src/report.py --input output/insights.json --metrics output/metrics.json --output output/weekly_report.md
```

2. Harness de avaliação
```bash
python evaluate.py --data data/events.csv --output evaluation_report.json --zones data/zones.json
```
 
Para avaliar com o dataset de validação:
```bash
python evaluate.py --data events_validation.csv --output evaluation_report.json --zones data/zones.json
```

## Outputs
Ao longo do trabalho, são gerados outputs que são guardados na pasta `outputs/`, onde cada um deles representa o seguinte:
1. **journeys.csv'**: reconstrói as trajetórias, onde faz uma linha por visita a uma zona
2. **metrics.json**: métricas calculadas em Python (tráfego, zonas, funil, anomalias)
3. **insights.json**: insights gerados pelo LLM (estratégias A e B com comparação)
4. **weekly_report.md**: briefing semanal para o gestor de loja
5. **evaluation_report.json**: ao correr o harness de avaliação, os resultados são guardados aqui
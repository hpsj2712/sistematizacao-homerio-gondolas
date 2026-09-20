# 🛒 Auditoria Inteligente de Gôndolas no Varejo via Visão Computacional

![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-blue)
![MobileSAM](https://img.shields.io/badge/Segmentação-MobileSAM-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)
![Kaggle](https://img.shields.io/badge/Environment-Kaggle%20GPU-orange)
![License](https://img.shields.io/badge/License-CC0%201.0-brightgreen)

> **Trabalho de Sistematização — Pós-Graduação em Visão Computacional e Reconhecimento de Padrões**  
> **Instituição:** CEUB (Centro de Ensino Unificado de Brasília)  
> **Orientador:** Prof. Romes Heriberto  

---

## 👥 Trabalho Individual

* **Homério Parreira da Silva Junior** — *Desenvolvimento, Pipeline e Modelagem*

---

## 📹 Vídeo-Pitch e Demonstração do Sistema

* 🔗 **Link do Vídeo (YouTube / Drive):** [Cole o Link Aqui]
* **Duração:** ~6 minutos
* **Conteúdo:** Apresentação da arquitetura, pipeline de dados, treino dos modelos e demonstração da inferência em tempo real com rastreamento de gôndola.

---

## 📌 1. Definição do Problema e Cenário de Aplicação

No setor varejista (supermercados e atacados), a monitoria contínua de prateleiras é crítica. A indisponibilidade de produtos nas prateleiras (*out-of-stock* ou rupturas) reduz vendas e afeta a experiência do consumidor.

### Objetivos do Projeto:
1. **Detecção e Classificação de Produtos**: Localizar e classificar itens dispostos em prateleiras.
2. **Identificação de Vazio/Ruptura**: Detectar lacunas e espaços vazios nas gôndolas em tempo real.
3. **Segmentação de Instâncias (Fase Avançada)**: Refinar a delineação de embalagens sobrepostas e garrafas usando máscaras do **SAM (Segment Anything Model)**.
4. **Rastreamento Dinâmico em Vídeo**: Aplicar rastreamento multi-objeto (**ByteTrack**) para contagem contínua durante navegação por vídeo na prateleira.

---

## 📊 2. Dataset e Análise Exploratória dos Dados (EDA)

O dataset utilizado foi derivado e curado a partir do *Retail Shelf Object Detection*, passando por um rigoroso processo de **Pruning de Qualidade** e reanotação geométrica automática.

### Mapeamento das 4 Classes Estratégicas:
* **0 - `garrafa`**: Bebidas, óleos, produtos de limpeza altos e cilíndricos.
* **1 - `pacote`**: Embalagens flexíveis (biscoitos, pães, grãos).
* **2 - `fardo_caixa`**: Caixas de papelão e fardos de produtos.
* **3 - `vazio`**: Espaços de ruptura na gôndola.

### Metodologia de Tratamento dos Dados:
1. **Filtro de Nitidez (Laplaciano)**: Seleção das **TOP 450 imagens** de maior qualidade com base na pontuação:
   $$\text{Score} = \text{Variância(Laplaciano)} \times (1 + 0.05 \times \text{Nº de Objetos})$$
2. **Divisão Rígida (Splits)**: 70% Treino | 15% Validação | 15% Teste (Inédito).
3. **Geração de Máscaras com SAM**: Utilização do **MobileSAM** para conversão de Bounding Boxes em polígonos de segmentação binária.

🔗 **Link do Dataset Oficial no Kaggle:** [homeriojr/sistematizacao-visao-computacional](https://www.kaggle.com/datasets/homeriojr/sistematizacao-visao-computacional)  
🏋️ **Link dos Pesos Treinados no Kaggle:** [homeriojr/sistematizacao-visao-yolo11-retail-shelf-weights](https://www.kaggle.com/datasets/homeriojr/sistematizacao-visao-yolo11-retail-shelf-weights)

---

## ⚙️ 3. Metodologia, Arquitetura e Hiperparâmetros

| Parâmetro | Configuração Detecção | Configuração Segmentação |
|---|---|---|
| **Modelo Base** | YOLO11 Nano (`yolo11n.pt`) | YOLO11 Nano Segmentação (`yolo11n-seg.pt`) |
| **Épocas** | 40 | 40 |
| **Tamanho da Imagem (`imgsz`)** | 640x640 | 640x640 |
| **Batch Size** | 16 | 16 |
| **Otimizador** | Auto (AdamW / SGD) | Auto (AdamW / SGD) |
| **Seed de Reprodutibilidade** | 42 | 42 |
| **Hardware** | Kaggle GPU NVIDIA P100 / T4 | Kaggle GPU NVIDIA P100 / T4 |

---

## 🏆 4. Resultados e Métricas (Conjunto de Teste)

Avaliação realizada no conjunto de teste inédito (nunca visto pelo modelo durante o treinamento):

| Modelo / Tarefa | mAP@0.5 | mAP@0.5:0.95 (IoU) | Precisão | Recall |
|---|---|---|---|---|
| **YOLO11 Detecção (Box)** | 0.3784 | 0.2195 | 0.5493 | 0.3685 |
| **YOLO11 Segmentação (Box)** | **0.5080** | **0.3381** | **0.5438** | **0.4899** |
| **YOLO11 Segmentação (Mask)** | **0.4831** | **0.2553** | **0.5263** | **0.4722** |

### Insights Principais:
- O aprendizado multitarefa da **Segmentação impulsionou a precisão da caixa**, aumentando o $mAP@0.5$ de **37,84%** para **50,80%**.
- O modelo apresentou desempenho sólido na identificação e delimitação exata de pacotes e garrafas.

---

## 🎨 5. Comparativo Visual e Análise de Erros

* **Ganhos da Segmentação sobre a Detecção Pura**: A segmentação reduz o ruído de fundo (*whitespace*) presente em caixas retangulares tradicionais, delineando garrafas cilíndricas com precisão no contorno real.
* **Análise de Erros**:
  * *Falsos Positivos*: Ocorrem predominantemente em produtos muito compactados e sobrepostos nos cantos inferiores da gôndola.
  * *Falsos Negativos*: Produtos no fundo das prateleiras com iluminação reduzida foram eventualmente omitidos.

---

## 🛠️ 6. Como Reproduzir este Projeto (Passo a Passo)

### Pré-requisitos:
Acesso ao Kaggle

### Execução do Pipeline no Kaggle/Colab:

* Abra os Notebooks no ambiente Kaggle GPU.

* Certifique-se de montar o dataset homeriojr/sistematizacao-visao-computacional.

* Execute o Notebook de Pré-processamento e Geração de Máscaras via SAM.

* Execute o Notebook de Treinamento e Avaliação.

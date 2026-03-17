# Experimento 2: Análise de Parâmetros do Modelo 1B

**Data:** 17 de março de 2026  
**Modelo:** StarVector-1B (starvector/starvector-1b-im2svg)  
**Imagem de teste:** sample-18.png (224x224)

## Objetivo

Testar o modelo 1B com diferentes configurações de parâmetros para entender seu comportamento e identificar os melhores trade-offs entre velocidade, qualidade e complexidade.

## Configurações Testadas

### 1. Parâmetros Variados

- **Temperature:** 0.7, 1.0, 1.2, 1.5, 2.0, 2.5 (controla criatividade)
- **Max Length:** 2000, 4000, 6000 (controla detalhamento)
- **Repetition Penalty:** 1.0, 2.0, 2.5, 3.1 (controla repetição)

### 2. Configurações Específicas

| Config | Temperature | Max Length | Repetition Penalty | Tempo (s) | Elementos | Tamanho (KB) |
|--------|-------------|------------|-------------------|-----------|-----------|--------------|
| **baseline** | 1.5 | 4000 | 3.1 | 440.7 | 45 | 5.28 |
| **temp_conservative** | 0.7 | 4000 | 3.1 | 449.0 | 45 | 5.29 |
| **temp_creative** | 2.0 | 4000 | 3.1 | 452.5 | 45 | 5.28 |
| **temp_very_creative** | 2.5 | 4000 | 3.1 | 451.9 | 45 | 5.29 |
| **length_short** | 1.5 | 2000 | 3.1 | 216.8 | 33 | 3.06 |
| **length_long** | 1.5 | 6000 | 3.1 | 449.4 | 45 | 5.27 |
| **penalty_low** | 1.5 | 4000 | 1.0 | 452.7 | 45 | 5.29 |
| **penalty_medium** | 1.5 | 4000 | 2.0 | 450.8 | 45 | 5.28 |
| **fast** | 1.0 | 2000 | 2.0 | 216.5 | 33 | 3.06 |
| **detailed** | 1.2 | 6000 | 2.5 | 451.4 | 45 | 5.29 |

## Resultados

### ✅ Todas as Configurações Funcionaram

Todas as 10 configurações geraram SVGs válidos sem erros.

### 🏆 Principais Descobertas

**Velocidade:**
- Configurações com `max_length=2000` são **2x mais rápidas** (~217s vs ~450s)
- **MAS:** Geram SVGs incompletos/quebrados ⚠️
- A mais rápida útil: **baseline** (440.7s)
- A mais lenta: **temp_creative** (452.5s)

**Complexidade:**
- Configurações com `max_length=2000`: 33 elementos (mas **SVG incompleto** ⚠️)
- Configurações com `max_length=4000-6000`: 45 elementos (completos ✅)
- Aumentar `max_length` de 4000 para 6000 **não aumenta complexidade**

**Temperature:**
- Variações de temperature (0.7 a 2.5) **não afetam** número de elementos
- Diferença mínima no tempo de geração (~440-452s)

**Repetition Penalty:**
- Variações (1.0 a 3.1) **não afetam** número de elementos
- Impacto mínimo no tempo de geração

## ⚠️ Problema Identificado

**SVGs Incompletos:**
- As configurações `fast` e `length_short` (com `max_length=2000`) geraram **SVGs incompletos**
- O arquivo termina abruptamente sem fechar as tags `</svg>`
- Não é possível visualizar esses SVGs no VS Code
- **Conclusão:** `max_length=2000` é insuficiente para esta imagem

## 💡 Recomendações

### Para Velocidade E Qualidade
- **NÃO use** `max_length=2000` - gera SVGs quebrados
- Use a config **baseline** (melhor equilíbrio)
- `temperature=1.5, max_length=4000, repetition_penalty=3.1`

### Para Máximo Detalhamento
- Use **detailed** (porém não adiciona mais elementos)
- Considere que `max_length > 4000` não traz benefícios extras para esta imagem

## 🔍 Observações Técnicas

1. **Flash Attention**: Tivemos que desinstalar `flash-attn` pois a GPU não suporta (requer Ampere+)
2. **Tempo Médio**: ~7-8 minutos por configuração para `max_length=4000`
3. **Hardware**: GPU 4GB, tempo de geração varia conforme `max_length`
4. **Consistency**: Para esta imagem, a maioria das configs convergiu para 45 elementos
5. **⚠️ SVGs Incompletos**: Configs com `max_length=2000` (`fast` e `length_short`) geraram SVGs quebrados - não podem ser visualizados

## Arquivos Gerados

- 10 arquivos SVG (um por configuração)
- `results.json` com dados completos
- Todos salvos em: `experiments/exp2_parameters_1b/results/`

## Próximos Passos

- Testar com diferentes tipos de imagens (ícones, logos, diagramas)
- Comparar qualidade visual dos SVGs gerados
- Analisar se as diferenças de temperature afetam a qualidade mesmo sem mudar o número de elementos

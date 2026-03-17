# 🧪 Experimentos com StarVector

Esta pasta contém experimentos estruturados para avaliar o desempenho e qualidade do StarVector.

## 📁 Estrutura

```
experiments/
├── exp1_model_comparison/     # Comparação 1B vs 8B
├── exp2_image_types/          # Testes com diferentes tipos de imagens
├── exp3_parameters/           # Impacto dos parâmetros de geração
├── exp4_metrics/              # Análise de métricas de qualidade
├── exp5_batch_processing/     # Processamento em lote
├── images/                    # Imagens de teste
└── results/                   # Resultados dos experimentos
```

## 🔬 Experimentos Disponíveis

### Experimento 1: Comparação de Modelos
**Objetivo:** Comparar StarVector-1B vs StarVector-8B

**Como executar:**
```bash
python experiments/exp1_model_comparison/exp1_model_comparison.py --image path/to/image.png
```

**Métricas avaliadas:**
- ⏱️ Tempo de carregamento do modelo
- ⚡ Tempo de geração do SVG
- 📏 Complexidade do SVG (número de elementos, tamanho)
- 🎨 Qualidade visual (comparação manual)

**Resultados esperados:**
- Arquivos SVG gerados por cada modelo
- Renderizações PNG para comparação visual
- JSON com métricas comparativas

---

### Experimento 2: Tipos de Imagens (Em desenvolvimento)
**Objetivo:** Avaliar performance em diferentes tipos de imagens

**Tipos testados:**
- Ícones simples (monocromáticos)
- Logos coloridos
- Diagramas técnicos
- Gráficos e charts

---

### Experimento 3: Parâmetros (Em desenvolvimento)
**Objetivo:** Entender o impacto dos parâmetros de geração

**Parâmetros testados:**
- `temperature` (0.5, 1.0, 1.5, 2.0)
- `max_length` (1000, 2000, 4000, 8000)
- `repetition_penalty` (1.0, 2.0, 3.1, 4.0)

---

### Experimento 4: Métricas de Qualidade (Em desenvolvimento)
**Objetivo:** Avaliar qualidade usando métricas objetivas

**Métricas:**
- SSIM (Structural Similarity)
- LPIPS (Learned Perceptual Image Patch Similarity)
- MSE (Mean Squared Error)
- Tempo de geração

---

### Experimento 5: Batch Processing (Em desenvolvimento)
**Objetivo:** Avaliar eficiência no processamento de múltiplas imagens

---

## 📊 Como Adicionar Imagens de Teste

Coloque suas imagens na pasta `experiments/images/`:

```bash
cp sua_imagem.png experiments/images/
```

**Tipos de imagens recomendadas:**
- ✅ Ícones e logos (funciona melhor)
- ✅ Diagramas técnicos
- ✅ Gráficos simples
- ❌ Fotografias realistas (não funciona bem)
- ❌ Imagens muito complexas

## 📈 Visualização de Resultados

Os resultados são salvos em:
- `experiments/results/exp{N}/` - Arquivos gerados
- `experiments/results/exp{N}/*_results.json` - Métricas em JSON

Para comparar visualmente:
1. Abra os arquivos PNG gerados
2. Compare com a imagem original
3. Analise o código SVG gerado

## 🎯 Próximos Passos

- [ ] Implementar Experimento 2 (Tipos de Imagens)
- [ ] Implementar Experimento 3 (Parâmetros)
- [ ] Implementar Experimento 4 (Métricas)
- [ ] Implementar Experimento 5 (Batch Processing)
- [ ] Criar script de análise agregada de resultados
- [ ] Gerar relatório final com gráficos

## 📝 Notas

- Todos os experimentos usam a mesma seed quando possível para reprodutibilidade
- Resultados são salvos automaticamente em JSON para análise posterior
- Certifique-se de ter espaço em disco (modelos são grandes ~5GB cada)

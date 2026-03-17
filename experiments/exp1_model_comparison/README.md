# Experimento 1: Comparação de Modelos

## Objetivo
Comparar o desempenho dos modelos StarVector 1B e 8B na geração de SVG a partir de imagens.

## Limitações de Hardware

**Hardware Atual:**
- GPU: 4GB VRAM
- RAM: ~16GB
- CUDA: 12.4

**Resultados da Tentativa de Carregamento do Modelo 8B:**

### Teste 1: Modelo sem quantização
- ❌ **Falha**: `torch.cuda.OutOfMemoryError: CUDA out of memory`
- Memória necessária: ~15GB
- Memória disponível: 4GB

### Teste 2: Quantização 8-bit
- ❌ **Falha**: `ValueError: Some modules are dispatched on the CPU or the disk`
- Memória reduzida para ~7.5GB, ainda insuficiente

### Teste 3: Quantização 4-bit com CPU Offload
- ❌ **Falha**: `RuntimeError: DefaultCPUAllocator: can't allocate memory`
- Tentou alocar 14GB na CPU durante o carregamento
- Sistema não tem memória suficiente para o processo de quantização

## Conclusão

O modelo 8B requer **no mínimo**:
- 8GB VRAM para quantização 8-bit
- 6GB VRAM para quantização 4-bit
- ~20GB RAM total do sistema

**Modelo 1B**:
- ✅ Funciona perfeitamente com 4GB VRAM
- Tempo de carregamento: ~5-10 segundos
- Geração de SVG: ~2-5 segundos por imagem
- Qualidade: Excelente para icons, logos e diagramas simples

## Alternativas para o Experimento

### Opção 1: Focar no modelo 1B (Recomendado)
Fazer análise detalhada do 1B:
- Diferentes tipos de imagens (ícones, logos, diagramas, emoji)
- Variação de parâmetros (temperature, max_length)
- Métricas de qualidade (complexidade SVG, tempo de geração)
- Diferentes casos de uso

### Opção 2: Usar serviços em nuvem
- Google Colab (GPU T4 com 15GB)
- Kaggle Notebooks (GPU P100 com 16GB)
- AWS/Azure/GCP com instâncias GPU

### Opção 3: Comparação qualitativa documentada
- Usar exemplos públicos de outputs do 8B
- Comparar com resultados do 1B
- Análise teórica baseada na arquitetura

## Próximos Passos

1. Executar experimento completo com modelo 1B
2. Documentar resultados e métricas
3. Incluir nota sobre limitações de hardware no relatório
4. Considerar acesso a GPU maior para trabalhos futuros

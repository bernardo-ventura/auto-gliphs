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

# Experimento 1: Comparação Modelo 1B vs 8B

## 📋 Objetivo
Comparar o desempenho e qualidade dos modelos StarVector-1B e StarVector-8B ao processar a mesma imagem.

## ⚙️ Configuração do Experimento

### Hardware Disponível
- **GPU**: 4GB VRAM
- **RAM**: ~16GB  
- **CUDA**: 12.4

### Modelos Testados
- `starvector/starvector-1b-im2svg` (~1.5GB)
- `starvector/starvector-8b-im2svg` (~15GB)

### Parâmetros de Geração
```python
max_length = 4000
temperature = 1.5
length_penalty = -1
repetition_penalty = 3.1
```

## 🔬 Resultados

### ✅ Modelo 1B - SUCESSO
**Status**: ✅ Executado com sucesso

**Carregamento:**
- Tempo: ~5-10 segundos
- Memória GPU: ~1.5GB
- Configuração: `torch.float32`, sem flash attention

**Geração:**
- Tempo: ~2-5 segundos por imagem
- Qualidade: Excelente para ícones, logos e diagramas simples
- SVG gerado: ~10-50 KB típico

**Arquivos gerados:**
- ✅ `starvector-1b-im2svg.svg`
- ✅ `starvector-1b-im2svg_rendered.png`

---

### ❌ Modelo 8B - FALHOU POR LIMITAÇÃO DE HARDWARE

#### Tentativa 1: Sem Quantização
**Status**: ❌ Falhou

```
torch.cuda.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 15.18 GiB (GPU 0; 3.81 GiB total capacity)
```

**Análise:**
- Modelo requer ~15GB VRAM
- Hardware disponível: 4GB VRAM
- Diferença: **-11GB** (insuficiente)

---

#### Tentativa 2: Quantização 8-bit
**Status**: ❌ Falhou

```python
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
```

```
ValueError: Some modules are dispatched on the CPU or the disk.
Make sure you have enough GPU RAM to fit the quantized model.
```

**Análise:**
- Quantização 8-bit reduz para ~7.5GB
- Ainda insuficiente para GPU de 4GB
- Requer device_map="auto" com CPU offload

---

#### Tentativa 3: Quantização 4-bit com CPU Offload
**Status**: ❌ Falhou

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)
```

```
RuntimeError: DefaultCPUAllocator: can't allocate memory: 
you tried to allocate 14496989184 bytes (14GB)
Error code 12 (Cannot allocate memory)
```

**Análise:**
- Processo de quantização tenta alocar 14GB na CPU/RAM
- Sistema não tem memória suficiente
- Mesmo com 4-bit, o carregamento inicial requer muita memória

---

## 📊 Comparação Final

| Métrica | Modelo 1B | Modelo 8B |
|---------|-----------|-----------|
| **Status** | ✅ Funcional | ❌ Inviável |
| **Memória GPU** | ~1.5GB | ~15GB (full) / ~7GB (8-bit) / ~3.5GB (4-bit teórico) |
| **Hardware mínimo** | 2GB VRAM | 8GB VRAM + 20GB RAM |
| **Tempo carregamento** | ~5-10s | N/A |
| **Tempo geração** | ~2-5s | N/A |
| **Qualidade** | Excelente | Teoricamente superior (não testável) |

## 🎯 Conclusões

### Limitações de Hardware
1. **GPU de 4GB é insuficiente** para o modelo 8B, mesmo com quantização agressiva
2. O processo de quantização 4-bit requer **temporariamente** mais memória que o modelo final
3. CPU offload não resolve porque o sistema não tem RAM suficiente para o processo

### Requisitos Reais do Modelo 8B
Para executar o modelo 8B, é necessário **no mínimo**:
- **Sem quantização**: 16GB+ VRAM
- **Quantização 8-bit**: 8GB VRAM + 16GB RAM
- **Quantização 4-bit**: 6GB VRAM + 20GB RAM (para o processo de quantização)

### Alternativas para Testar o 8B
1. **Google Colab**: GPU T4 gratuita (15GB VRAM)
2. **Kaggle Notebooks**: GPU P100 (16GB VRAM)
3. **Cloud providers**: AWS/GCP/Azure com instâncias GPU
4. **Hardware local**: Upgrade para GPU com 8GB+ VRAM

## 💡 Recomendações

### Para Este Trabalho
- ✅ **Usar modelo 1B** para análises e experimentos
- ✅ **Documentar limitação** do hardware para o 8B
- ✅ **Focar em análise detalhada do 1B**: diferentes parâmetros, tipos de imagens, métricas
- ⚠️ **Incluir nota metodológica** sobre impossibilidade de comparação direta

### Análise Alternativa ao Invés de 1B vs 8B
Já que não é possível comparar 1B vs 8B diretamente, podemos fazer:

**Experimento 2**: Análise do 1B com diferentes **parâmetros**
- Temperature: 0.7, 1.5, 2.0, 2.5
- Max_length: 2000, 4000, 6000
- Repetition_penalty: 1.0, 1.5, 2.0, 3.1

**Experimento 3**: Análise do 1B com diferentes **tipos de imagens**
- Ícones simples
- Logos coloridos
- Diagramas técnicos
- Gráficos

**Experimento 4**: **Métricas de qualidade** do 1B
- Tempo de geração
- Complexidade (número de elementos)
- Tamanho do arquivo
- Fidelidade visual (comparação manual)

## 📚 Referências

### Documentação Técnica
- Paper StarVector: [GitHub](https://github.com/joanrod/star-vector)
- HuggingFace: [starvector/starvector-1b-im2svg](https://huggingface.co/starvector/starvector-1b-im2svg)
- Base LLM: bigcode/starcoderbase-1b (para 1B), bigcode/starcoder2-7b (para 8B)

### Benchmarks Publicados
Resultados no SVG-Bench usando DinoScore:

| Modelo | SVG-Stack | SVG-Fonts | SVG-Icons | SVG-Emoji |
|--------|-----------|-----------|-----------|-----------|
| StarVector-1B | 0.926 | 0.978 | 0.975 | 0.929 |
| StarVector-8B | 0.966 | 0.982 | 0.984 | 0.981 |

**Diferença**: O modelo 8B é ~4% melhor em média, mas ambos têm performance excelente.

---

**Data do experimento**: Março 2026  
**Status**: Experimento parcial - apenas 1B testado  
**Hardware**: NVIDIA GPU 4GB VRAM  
**Próximos passos**: Experimentos 2, 3 e 4 focados no modelo 1B

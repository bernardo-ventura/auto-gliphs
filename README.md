# StarVector + SAM2

Projeto de pesquisa combinando [StarVector](https://github.com/joanrod/star-vector) (geração de SVG a partir de imagens) e [SAM2](https://github.com/facebookresearch/sam2) (segmentação semântica) para vetorização avançada de imagens.

**Tecnologias:**
- StarVector: Modelos baseados em transformers para gerar código SVG
- SAM2: Segmentação automática de objetos em imagens
- PyTorch 2.5.1 | Transformers 4.49.0

---

## 🚀 Setup

### **Pré-requisitos**
- Miniconda/Anaconda ([Download](https://docs.conda.io/en/latest/miniconda.html))
- GPU NVIDIA com CUDA (opcional, recomendado)

### **Instalação**
```bash
# 1. Clonar repositório
git clone <url-do-repositorio>
cd star-vector

# 2. Criar e ativar ambiente
conda env create -f environment.yml
conda activate starvector

# 3. Instalar dependências
pip install -e .

# 4. Autenticar no HuggingFace
huggingface-cli login
```

**HuggingFace Setup:**
- Token: https://huggingface.co/settings/tokens (tipo: Read)
- Aceitar termos: https://huggingface.co/bigcode/starcoderbase-1b

---

## ✅ Testar Instalação

### **StarVector (Image → SVG)**
```bash
python test_image2svg.py
```
Gera `output.svg` a partir de uma imagem de exemplo.

### **SAM2 (Segmentação)**
```bash
python experiments_sam2/exp_multiple_images/exp_multiple_images.py
```
Segmenta imagens em `assets/examples/` e salva resultados.

---

## 📁 Estrutura do Projeto

```
star-vector/
├── experiments_starvector/    # Experimentos com StarVector
├── experiments_sam2/          # Experimentos com SAM2
├── starvector/                # Código fonte StarVector
├── assets/examples/           # Imagens de exemplo
├── test_image2svg.py          # Script de teste rápido
└── environment.yml            # Configuração do ambiente
```

---

## ⚙️ Notas Técnicas

- Python 3.11 via Conda
- Primeira execução baixa ~5GB de modelos
- `flash_attn` desabilitado (requer CUDA toolkit completo)
- GPU recomendada mas não obrigatória

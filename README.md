# StarVector + SAM2 - Image to SVG Vectorization

Este é um fork do projeto [StarVector](https://github.com/joanrod/star-vector) com integração do SAM2 (Segment Anything Model 2) para geração de SVG com anotações semânticas.

## 📦 Installation / Instalação

**Quick Start:**
```bash
pip install -r requirements.txt
pip install -e .
```

**Detailed instructions:** See [INSTALLATION.md](INSTALLATION.md)

**Available requirements files:**
- `requirements.txt` - Full installation (recommended)
- `requirements-minimal.txt` - Inference only (lightweight)

---

## 🚀 Instalação Realizada

### Ambiente
- Python 3.11.3 (via Conda)
- Ambiente: `starvector`
- GPU: CUDA (se disponível)

### Dependências Principais
- PyTorch 2.5.1
- Transformers 4.49.0
- Bibliotecas SVG (cairosvg, svgpathtools, svglib)

### Instalação
```bash
# Criar ambiente conda
conda create -n starvector python=3.11.3 -y
conda activate starvector

# Instalar dependências do sistema (Ubuntu/Debian)
sudo apt-get install libcairo2-dev pkg-config python3-dev

# Instalar o pacote
pip install --upgrade pip
pip install -e .
```

### Autenticação HuggingFace
É necessário fazer login no HuggingFace e aceitar os termos do modelo StarCoder:

```bash
huggingface-cli login
```

Depois acesse: https://huggingface.co/bigcode/starcoderbase-1b e aceite os termos.

## 📝 Scripts Personalizados

### Image2SVG - Gerar SVG a partir de Imagem

Script: `test_image2svg.py`

```bash
python test_image2svg.py
```

**O que faz:**
- Carrega modelo `starvector-1b-im2svg`
- Processa imagem de exemplo
- Gera código SVG vetorizado
- Salva resultado em `output.svg` e `output_rendered.png`

**Parâmetros principais:**
- `max_length=4000` - Tamanho máximo do SVG gerado
- `temperature=1.5` - Criatividade na geração
- `length_penalty=-1` - Penalidade por comprimento
- `repetition_penalty=3.1` - Evita repetições

### Exemplo de Uso

```python
from PIL import Image
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.util import process_and_rasterize_svg
import torch

# Carregar modelo
model = StarVectorForCausalLM.from_pretrained("starvector/starvector-1b-im2svg", torch_dtype=torch.float32)
model.cuda()
model.eval()

# Processar imagem
image = Image.open('sua_imagem.png').convert('RGB')
image_tensor = model.process_images([image])[0].to(torch.float16).cuda()

# Gerar SVG
batch = {"image": image_tensor}
raw_svg = model.generate_im2svg(batch, max_length=4000, temperature=1.5, length_penalty=-1, repetition_penalty=3.1)[0]

# Processar resultado
svg, raster_image = process_and_rasterize_svg(raw_svg)

# Salvar
with open("output.svg", "w") as f:
    f.write(svg)
```

## 📊 Modelos Disponíveis

- ✅ `starvector/starvector-1b-im2svg` - Image → SVG (1B parâmetros)
- ✅ `starvector/starvector-8b-im2svg` - Image → SVG (8B parâmetros, mais preciso)
- ❌ Text2SVG - Requer treinamento/fine-tuning

## 🔧 Modificações Realizadas

1. **pyproject.toml**: Comentado `flash_attn` (requer CUDA toolkit completo)
2. **Scripts de teste**: Criados para facilitar uso dos modelos
3. **.gitignore**: Atualizado para ignorar outputs gerados


## 🎯 Próximos Passos

- [ ] Testar modelo 8B (maior e mais preciso)
- [ ] Testar outras imagens
- [ ] Treinar modelo Text2SVG



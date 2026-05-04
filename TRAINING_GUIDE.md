# Guia de Treinamento - Text2SVG StarVector

## Ambiente Configurado

**Hardware**: 2x NVIDIA RTX 3080 Ti (12GB cada)  
**Software**: Python 3.11.3, PyTorch 2.5.1, CUDA 12.4, DeepSpeed 0.18.9, Transformers 4.49.0  
**Dataset**: starvector/text2svg-stack (2.17M amostras treino, 5.7K teste)

## Como Treinar

### Configurar Token HuggingFace

**Opção 1 (Recomendado)**: Criar arquivo `.env`
```bash
cp .env.example .env
# Editar .env e adicionar seu token
```

**Opção 2**: Export manual
```bash
export HF_TOKEN=seu_token_huggingface_aqui
```

### Verificar GPUs Disponíveis
```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### Com 2 GPUs Livres (Recomendado)
```bash
cd /home/bernardo/projetos/auto-gliphs
conda activate starvector
tmux new -s starvector_train

# Dentro do tmux:
./scripts/train/train-starvector-1b-text2svg-2gpus.sh

# Sair sem parar: Ctrl+B depois D
# Voltar: tmux attach -t starvector_train
```

### Com 1 GPU (Limitado - OOM após 1 step)
```bash
cd /home/bernardo/projetos/auto-gliphs
conda activate starvector
tmux new -s starvector_train

export CUDA_VISIBLE_DEVICES=0
./scripts/train/train-starvector-1b-text2svg-custom.sh
```

**Nota**: 1 GPU de 12GB é insuficiente. O treinamento processa 1 step mas falha com Out of Memory. Use 2 GPUs com DeepSpeed ZeRO-2.

## Monitoramento

- Ver progresso: `tmux attach -t starvector_train`
- Ver GPUs: `nvidia-smi`
- Ver checkpoints: `ls -lh output/`

Checkpoints salvos em: `output/starvector/text2svg-stack/starvector-1b-text2svg/`

## Configuração

**Arquivo**: `configs/models/starvector-1b/text2svg-stack.yaml`

```yaml
model:
  max_length: 4096
  task: text2svg
  train_LLM: true

training:
  n_epochs: 2
  batch_size: 1
  gradient_accumulation_steps: 8

data:
  dataset_name: starvector/text2svg-stack
```

Batch size efetivo: 1 × 8 × 2 GPUs = 16 amostras/update

## Resumo das Alterações Feitas Hoje

### Problemas Corrigidos

1. **KeyError: HF_TOKEN**: Adicionado export HF_TOKEN nos scripts
2. **DeepSpeed batch_size**: Adicionado train_micro_batch_size_per_gpu nos configs
3. **Preparação do modelo**: Modificado train.py para preparar modelo+dataloaders juntos
4. **Forward incompatível**: Modificado starvector_arch.py para aceitar batch como argumento
5. **Parâmetros não treináveis**: Adicionado verificação e correção automática em train.py após accelerator.prepare()
6. **CUDA OOM**: Reduzido max_length de 8192 para 4096

### Arquivos Modificados

- `starvector/model/starvector_arch.py`
- `starvector/train/train.py`
- `configs/models/starvector-1b/text2svg-stack.yaml`
- `configs/accelerate/deepspeed-1-gpu.yaml`
- `configs/accelerate/deepspeed-2-gpu.yaml`
- `scripts/train/train-starvector-1b-text2svg-custom.sh`
- `scripts/train/train-starvector-1b-text2svg-2gpus.sh` (novo)

### Status Atual

- Ambiente completamente configurado
- Modelo carrega e processa forward pass
- Parâmetros treináveis configurados corretamente (1.1B params)
- Com 1 GPU: treina 1 step mas falha com OOM
- Com 2 GPUs + DeepSpeed ZeRO-2: deve funcionar (divide optimizer states)

### Requisitos de Memória

**1 GPU (12GB)**: Insuficiente
- Modelo: ~2GB
- Ativações: ~3-4GB  
- Gradientes: ~2GB
- Optimizer: ~4GB
- Total: ~11-12GB (sem margem)

**2 GPUs (24GB)**: Suficiente com DeepSpeed
- Por GPU: ~9-10GB
- Sobra: ~2-3GB por GPU

## Comandos Úteis

```bash
# Ver GPUs
nvidia-smi
watch -n 1 nvidia-smi

# Ver processos usando GPU
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Tmux
tmux ls                           # Listar sessões
tmux attach -t starvector_train   # Anexar
tmux kill-session -t nome         # Matar sessão

# Ver checkpoints
ls -lh output/

# Espaço em disco
df -h ~
```

#!/bin/bash

# ====================================
# Configurações para Treinar Text2SVG
# ====================================

# Carregar variáveis de ambiente do .env se existir
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Verificar se HF_TOKEN está configurado
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  ERRO: HF_TOKEN não configurado!"
    echo "Configure de uma das formas:"
    echo "  1. Crie arquivo .env (veja .env.example)"
    echo "  2. Export manual: export HF_TOKEN=seu_token"
    exit 1
fi

# Diretórios (AJUSTE CONFORME NECESSÁRIO)
export HF_HOME=${HF_HOME:-/home/bernardo/.cache/huggingface}
export OUTPUT_DIR=${OUTPUT_DIR:-/home/bernardo/projetos/auto-gliphs/output}

# GPU Configuration (ambiente compartilhado - usar apenas GPU 0)
export CUDA_VISIBLE_DEVICES=0  # Usar apenas GPU 0 (GPU 1 está ocupada)

# Wandb (opcional - para monitoramento de experimentos)
# Se quiser usar Wandb, descomente e adicione seu token:
# export WANDB_API_KEY=seu_token_wandb_aqui
# E mude use_wandb para true no config YAML

# ====================================
# Lançar Treinamento
# ====================================

echo "🚀 Iniciando treinamento Text2SVG com StarVector-1B"
echo "📊 Hardware: 1x RTX 3080 Ti (12GB) - GPU 0"
echo "⚠️  GPU 1 em uso por outro usuário"
echo "📁 Output: $OUTPUT_DIR"
echo ""

# Usar configuração para 1 GPU SEM DeepSpeed (mais estável)
accelerate launch --config_file configs/accelerate/1-gpu.yaml \
                starvector/train/train.py \
                config=configs/models/starvector-1b/text2svg-stack.yaml

echo ""
echo "✅ Treinamento concluído!"

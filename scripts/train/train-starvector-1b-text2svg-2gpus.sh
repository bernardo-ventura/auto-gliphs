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

# GPU Configuration - USAR AMBAS GPUs
# Certifique-se de que ambas estão livres antes de rodar!
# export CUDA_VISIBLE_DEVICES=0,1  # Descomentar se necessário especificar

# Wandb (opcional - para monitoramento de experimentos)
# Se quiser usar Wandb, descomente e adicione seu token:
# export WANDB_API_KEY=seu_token_wandb_aqui
# E mude use_wandb para true no config YAML

# ====================================
# Lançar Treinamento
# ====================================

echo "🚀 Iniciando treinamento Text2SVG com StarVector-1B"
echo "📊 Hardware: 2x RTX 3080 Ti (12GB cada) com DeepSpeed ZeRO-2"
echo "📁 Output: $OUTPUT_DIR"
echo ""

# Verificar se ambas GPUs estão livres
echo "🔍 Verificando GPUs disponíveis..."
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo ""
echo "⚠️  ATENÇÃO: Certifique-se de que ambas GPUs estão livres!"
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Usar configuração para 2 GPUs com DeepSpeed
accelerate launch --config_file configs/accelerate/deepspeed-2-gpu.yaml \
                starvector/train/train.py \
                config=configs/models/starvector-1b/text2svg-stack.yaml

echo ""
echo "✅ Treinamento concluído!"

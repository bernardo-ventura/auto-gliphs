#!/bin/bash
# Script para iniciar o treinamento Text2SVG em sessão tmux

set -e

echo "🚀 StarVector Text2SVG - Início do Treinamento"
echo "=============================================="
echo ""

# Verificar se sessão já existe
if tmux has-session -t starvector_train 2>/dev/null; then
    echo "⚠️  Sessão 'starvector_train' já existe!"
    echo ""
    echo "Opções:"
    echo "  1. Anexar à sessão existente: tmux attach -t starvector_train"
    echo "  2. Matar sessão antiga: tmux kill-session -t starvector_train"
    echo ""
    exit 1
fi

# Criar nova sessão tmux
echo "📺 Criando sessão tmux: starvector_train"
tmux new-session -d -s starvector_train

# Configurar ambiente na sessão
tmux send-keys -t starvector_train "conda activate starvector" C-m
sleep 1
tmux send-keys -t starvector_train "cd /home/bernardo/projetos/auto-gliphs" C-m
sleep 1
# Configure HF_TOKEN antes de rodar este script:
# export HF_TOKEN=seu_token_huggingface
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  ERRO: HF_TOKEN não configurado!"
    echo "Configure com: export HF_TOKEN=seu_token_huggingface"
    tmux kill-session -t starvector_train
    exit 1
fi
tmux send-keys -t starvector_train "export HF_TOKEN=$HF_TOKEN" C-m
sleep 1

# Opcional: Configurar Wandb (descomente se quiser usar)
# tmux send-keys -t starvector_train "export WANDB_API_KEY=sua_api_key_aqui" C-m

echo "✅ Sessão criada!"
echo ""
echo "🎯 Iniciando treinamento..."
tmux send-keys -t starvector_train "./scripts/train/train-starvector-1b-text2svg-custom.sh" C-m

echo ""
echo "✅ Treinamento iniciado em background!"
echo ""
echo "📋 Comandos úteis:"
echo "  • Ver sessão:      tmux attach -t starvector_train"
echo "  • Sair (Ctrl+B D): Detach sem matar o processo"
echo "  • Listar sessões:  tmux ls"
echo "  • Matar sessão:    tmux kill-session -t starvector_train"
echo ""
echo "📊 Monitorar GPUs:   watch -n 1 nvidia-smi"
echo "📁 Checkpoints:      ls -lh output/"
echo ""
echo "🔗 Anexando à sessão em 2 segundos..."
sleep 2

# Anexar à sessão
tmux attach -t starvector_train

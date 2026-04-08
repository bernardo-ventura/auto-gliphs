#!/usr/bin/env python3
"""
Experimento 2: Análise de Parâmetros do Modelo 1B

Este experimento foca em entender o comportamento do modelo 1B
testando diferentes configurações de parâmetros.

Parâmetros testados:
- temperature: 0.7, 1.5, 2.0, 2.5 (criatividade)
- max_length: 2000, 4000, 6000 (detalhamento)
- repetition_penalty: 1.0, 1.5, 2.0, 3.1 (anti-repetição)

Uso:
    python exp2_parameters_1b.py --image path/to/image.png
"""

import argparse
import time
import json
from pathlib import Path
from PIL import Image
import torch
from starvector.model.starvector_arch import StarVectorForCausalLM


def count_svg_elements(svg_text):
    """Conta elementos SVG"""
    elements = ['<path', '<line', '<rect', '<circle', '<ellipse', '<polygon', '<polyline']
    return sum(svg_text.count(elem) for elem in elements)


def get_svg_stats(svg_text):
    """Retorna estatísticas do SVG"""
    return {
        'num_elements': count_svg_elements(svg_text),
        'num_paths': svg_text.count('<path'),
        'file_size_kb': len(svg_text.encode('utf-8')) / 1024,
        'num_lines': len(svg_text.split('\n'))
    }


def test_configuration(model, batch, config_name, params):
    """Testa uma configuração específica de parâmetros"""
    
    print(f"\n{'─'*60}")
    print(f"🧪 Testando: {config_name}")
    print(f"   temperature={params['temperature']}, "
          f"max_length={params['max_length']}, "
          f"repetition_penalty={params['repetition_penalty']}")
    
    # Gerar SVG
    start_time = time.time()
    try:
        svg_text = model.generate_im2svg(
            batch,
            max_length=params['max_length'],
            temperature=params['temperature'],
            repetition_penalty=params['repetition_penalty'],
            length_penalty=-1
        )[0]
        
        generation_time = time.time() - start_time
        stats = get_svg_stats(svg_text)
        
        print(f"   ⏱️  Tempo: {generation_time:.2f}s")
        print(f"   📊 Elementos: {stats['num_elements']}")
        print(f"   💾 Tamanho: {stats['file_size_kb']:.2f} KB")
        print(f"   ✅ Sucesso")
        
        return {
            'config_name': config_name,
            'params': params,
            'generation_time': round(generation_time, 2),
            'stats': stats,
            'svg': svg_text,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return {
            'config_name': config_name,
            'params': params,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Testar parâmetros do modelo 1B')
    parser.add_argument('--image', type=str, required=True, 
                       help='Caminho para a imagem de entrada')
    parser.add_argument('--output-dir', type=str, 
                       default='experiments/exp2_parameters_1b/results',
                       help='Diretório para salvar os resultados')
    args = parser.parse_args()
    
    # Criar diretório
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("EXPERIMENTO 2: Análise de Parâmetros do Modelo 1B")
    print("="*60)
    
    # Carregar modelo
    print("\n📦 Carregando modelo StarVector-1B...")
    model = StarVectorForCausalLM.from_pretrained(
        "starvector/starvector-1b-im2svg",
        torch_dtype="auto"
    )
    model.cuda()
    model.eval()
    print("   ✅ Modelo carregado")
    
    # Carregar imagem
    print(f"\n📷 Carregando imagem: {args.image}")
    image_pil = Image.open(args.image).convert('RGB')
    print(f"   Dimensões: {image_pil.size}")
    
    # Processar imagem
    image = model.process_images([image_pil])[0].to(torch.float16).cuda()
    batch = {"image": image}
    
    # Configurações para testar
    configurations = [
        # Baseline (recomendado)
        {
            'name': 'baseline',
            'params': {'temperature': 1.5, 'max_length': 4000, 'repetition_penalty': 3.1}
        },
        
        # Variações de temperature (criatividade)
        {
            'name': 'temp_conservative',
            'params': {'temperature': 0.7, 'max_length': 4000, 'repetition_penalty': 3.1}
        },
        {
            'name': 'temp_creative',
            'params': {'temperature': 2.0, 'max_length': 4000, 'repetition_penalty': 3.1}
        },
        {
            'name': 'temp_very_creative',
            'params': {'temperature': 2.5, 'max_length': 4000, 'repetition_penalty': 3.1}
        },
        
        # Variações de max_length (detalhamento)
        {
            'name': 'length_short',
            'params': {'temperature': 1.5, 'max_length': 2000, 'repetition_penalty': 3.1}
        },
        {
            'name': 'length_long',
            'params': {'temperature': 1.5, 'max_length': 6000, 'repetition_penalty': 3.1}
        },
        
        # Variações de repetition_penalty
        {
            'name': 'penalty_low',
            'params': {'temperature': 1.5, 'max_length': 4000, 'repetition_penalty': 1.0}
        },
        {
            'name': 'penalty_medium',
            'params': {'temperature': 1.5, 'max_length': 4000, 'repetition_penalty': 2.0}
        },
        
        # Combinações especiais
        {
            'name': 'fast',
            'params': {'temperature': 1.0, 'max_length': 2000, 'repetition_penalty': 2.0}
        },
        {
            'name': 'detailed',
            'params': {'temperature': 1.2, 'max_length': 6000, 'repetition_penalty': 2.5}
        }
    ]
    
    # Executar testes
    print(f"\n🔬 Executando {len(configurations)} configurações...")
    results = []
    
    for config in configurations:
        result = test_configuration(
            model, batch, config['name'], config['params']
        )
        results.append(result)
        
        # Salvar arquivos se sucesso
        if result['success']:
            # SVG
            svg_path = output_dir / f"{config['name']}.svg"
            with open(svg_path, 'w') as f:
                f.write(result['svg'])
            
            # Remover dados pesados do resultado
            result.pop('svg', None)
    
    # Salvar resultados JSON
    results_json = output_dir / "results.json"
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Análise comparativa
    print(f"\n{'='*60}")
    print("📊 ANÁLISE COMPARATIVA")
    print(f"{'='*60}")
    
    successful = [r for r in results if r['success']]
    
    if successful:
        print(f"\n✅ Configurações bem-sucedidas: {len(successful)}/{len(results)}")
        
        # Tempo
        print(f"\n⏱️  TEMPO DE GERAÇÃO:")
        times = sorted(successful, key=lambda x: x['generation_time'])
        print(f"   Mais rápida: {times[0]['config_name']} ({times[0]['generation_time']}s)")
        print(f"   Mais lenta: {times[-1]['config_name']} ({times[-1]['generation_time']}s)")
        avg_time = sum(r['generation_time'] for r in successful) / len(successful)
        print(f"   Média: {avg_time:.2f}s")
        
        # Complexidade
        print(f"\n📊 COMPLEXIDADE (número de elementos):")
        elements = sorted(successful, key=lambda x: x['stats']['num_elements'])
        print(f"   Mais simples: {elements[0]['config_name']} ({elements[0]['stats']['num_elements']} elementos)")
        print(f"   Mais complexo: {elements[-1]['config_name']} ({elements[-1]['stats']['num_elements']} elementos)")
        
        # Tamanho
        print(f"\n💾 TAMANHO DO ARQUIVO:")
        sizes = sorted(successful, key=lambda x: x['stats']['file_size_kb'])
        print(f"   Menor: {sizes[0]['config_name']} ({sizes[0]['stats']['file_size_kb']:.2f} KB)")
        print(f"   Maior: {sizes[-1]['config_name']} ({sizes[-1]['stats']['file_size_kb']:.2f} KB)")
        
        # Recomendações
        print(f"\n🎯 RECOMENDAÇÕES:")
        print(f"   • Velocidade: {times[0]['config_name']}")
        print(f"   • Balanceado: baseline")
        print(f"   • Detalhado: {elements[-1]['config_name']}")
    
    print(f"\n💾 Resultados salvos em: {output_dir}")
    print(f"   - JSON: {results_json}")
    print(f"   - SVGs: {output_dir}/*.svg")
    print(f"   - PNGs: {output_dir}/*_rendered.png")
    
    print(f"\n💡 Compare visualmente os resultados para escolher a melhor configuração!")


if __name__ == "__main__":
    main()

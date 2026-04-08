#!/usr/bin/env python3
"""
Experiment 3: Image Type Analysis

Tests the 1B model (baseline configuration) with different types of images
to evaluate whether it generates valid and visually acceptable SVGs for images
of varying complexity.

Baseline configuration used:
- temperature=1.5
- max_length=6000
- repetition_penalty=3.1

Usage:
    # Single image
    python exp3_image_types.py --images path/to/image1.png
    
    # Multiple images
    python exp3_image_types.py --images img1.png img2.png img3.png
    
    # With custom description
    python exp3_image_types.py --images img1.png --description "Colorful logo"
"""

import argparse
import time
import json
from pathlib import Path
from PIL import Image
import torch
from starvector.model.starvector_arch import StarVectorForCausalLM


# Baseline configuration (adjusted for more complex images)
BASELINE_CONFIG = {
    'temperature': 1.5,
    'max_length': 6000,
    'repetition_penalty': 3.1,
    'length_penalty': -1
}


def count_svg_elements(svg_text):
    """Counts SVG elements"""
    elements = ['<path', '<line', '<rect', '<circle', '<ellipse', '<polygon', '<polyline']
    return sum(svg_text.count(elem) for elem in elements)


def check_svg_valid(svg_text):
    """Checks if SVG is complete (has closing tag)"""
    return svg_text.strip().endswith('</svg>') or svg_text.strip().endswith('</g>')


def get_svg_stats(svg_text):
    """Returns SVG statistics"""
    is_valid = check_svg_valid(svg_text)
    return {
        'num_elements': count_svg_elements(svg_text),
        'num_paths': svg_text.count('<path'),
        'file_size_kb': len(svg_text.encode('utf-8')) / 1024,
        'num_lines': len(svg_text.split('\n')),
        'is_valid': is_valid,
        'has_closing_tag': '</svg>' in svg_text
    }


def process_image(model, image_path, description=None):
    """Processes an image and returns results"""
    
    image_path = Path(image_path)
    image_name = image_path.stem
    
    print(f"\n{'='*60}")
    print(f"📷 Processando: {image_path.name}")
    if description:
        print(f"   Tipo: {description}")
    
    # Carregar imagem
    try:
        image_pil = Image.open(image_path).convert('RGB')
        print(f"   Dimensões: {image_pil.size}")
    except Exception as e:
        print(f"   ❌ Erro ao carregar imagem: {e}")
        return {
            'image_name': image_name,
            'image_path': str(image_path),
            'description': description,
            'success': False,
            'error': f'Failed to load image: {e}'
        }
    
    # Processar imagem
    try:
        image = model.process_images([image_pil])[0].to(torch.float16).cuda()
        batch = {"image": image}
    except Exception as e:
        print(f"   ❌ Erro ao processar imagem: {e}")
        return {
            'image_name': image_name,
            'image_path': str(image_path),
            'description': description,
            'success': False,
            'error': f'Failed to process image: {e}'
        }
    
    # Gerar SVG
    print(f"   🔄 Gerando SVG (baseline config)...")
    start_time = time.time()
    
    try:
        svg_text = model.generate_im2svg(
            batch,
            **BASELINE_CONFIG
        )[0]
        
        generation_time = time.time() - start_time
        stats = get_svg_stats(svg_text)
        
        # Status
        if stats['is_valid']:
            print(f"   ✅ SVG válido gerado")
        else:
            print(f"   ⚠️  SVG incompleto (pode não renderizar)")
        
        print(f"   ⏱️  Tempo: {generation_time:.2f}s")
        print(f"   📊 Elementos: {stats['num_elements']}")
        print(f"   💾 Tamanho: {stats['file_size_kb']:.2f} KB")
        
        return {
            'image_name': image_name,
            'image_path': str(image_path),
            'image_size': image_pil.size,
            'description': description,
            'generation_time': round(generation_time, 2),
            'stats': stats,
            'svg': svg_text,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ Erro na geração: {e}")
        return {
            'image_name': image_name,
            'image_path': str(image_path),
            'description': description,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Experimento 3: Análise de Tipos de Imagens')
    parser.add_argument('--images', type=str, nargs='+', required=True,
                       help='Caminhos das imagens para processar')
    parser.add_argument('--descriptions', type=str, nargs='*',
                       help='Descrições das imagens (opcional, mesma ordem)')
    parser.add_argument('--output-dir', type=str,
                       default='experiments/exp3_image_types/results',
                       help='Diretório para salvar resultados')
    args = parser.parse_args()
    
    # Validar
    if args.descriptions and len(args.descriptions) != len(args.images):
        print("❌ Número de descrições deve ser igual ao número de imagens")
        return
    
    # Criar diretório
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("EXPERIMENTO 3: Análise de Tipos de Imagens")
    print("="*60)
    print(f"\n📋 Configuração Baseline:")
    print(f"   temperature: {BASELINE_CONFIG['temperature']}")
    print(f"   max_length: {BASELINE_CONFIG['max_length']}")
    print(f"   repetition_penalty: {BASELINE_CONFIG['repetition_penalty']}")
    print(f"\n📂 {len(args.images)} imagem(ns) para processar")
    
    # Carregar modelo
    print("\n📦 Carregando modelo StarVector-1B...")
    model = StarVectorForCausalLM.from_pretrained(
        "starvector/starvector-1b-im2svg",
        torch_dtype="auto"
    )
    model.cuda()
    model.eval()
    print("   ✅ Modelo carregado")
    
    # Processar cada imagem
    results = []
    descriptions = args.descriptions or [None] * len(args.images)
    
    for image_path, description in zip(args.images, descriptions):
        result = process_image(model, image_path, description)
        results.append(result)
        
        # Salvar SVG se sucesso
        if result['success']:
            svg_path = output_dir / f"{result['image_name']}.svg"
            with open(svg_path, 'w') as f:
                f.write(result['svg'])
            
            # Remover SVG do resultado JSON (muito grande)
            result_copy = result.copy()
            result_copy.pop('svg', None)
            result = result_copy
    
    # Salvar resultados JSON
    results_json = output_dir / "results.json"
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Análise comparativa
    print(f"\n{'='*60}")
    print("📊 ANÁLISE COMPARATIVA")
    print(f"{'='*60}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n✅ Sucessos: {len(successful)}/{len(results)}")
    if failed:
        print(f"❌ Falhas: {len(failed)}")
        for f in failed:
            print(f"   - {f['image_name']}: {f.get('error', 'Unknown')}")
    
    if successful:
        print(f"\n⏱️  TEMPO DE GERAÇÃO:")
        times = sorted(successful, key=lambda x: x['generation_time'])
        print(f"   Mais rápida: {times[0]['image_name']} ({times[0]['generation_time']}s)")
        print(f"   Mais lenta: {times[-1]['image_name']} ({times[-1]['generation_time']}s)")
        avg_time = sum(r['generation_time'] for r in successful) / len(successful)
        print(f"   Média: {avg_time:.2f}s")
        
        print(f"\n📊 COMPLEXIDADE:")
        by_elements = sorted(successful, key=lambda x: x['stats']['num_elements'])
        print(f"   Mais simples: {by_elements[0]['image_name']} ({by_elements[0]['stats']['num_elements']} elementos)")
        print(f"   Mais complexa: {by_elements[-1]['image_name']} ({by_elements[-1]['stats']['num_elements']} elementos)")
        
        print(f"\n✅ VALIDADE DOS SVGs:")
        valid = [r for r in successful if r['stats']['is_valid']]
        invalid = [r for r in successful if not r['stats']['is_valid']]
        print(f"   Válidos: {len(valid)}/{len(successful)}")
        if invalid:
            print(f"   ⚠️  Incompletos:")
            for inv in invalid:
                print(f"      - {inv['image_name']}")
        
        print(f"\n📋 DETALHES POR IMAGEM:")
        for r in successful:
            status = "✅" if r['stats']['is_valid'] else "⚠️"
            desc = f" ({r['description']})" if r['description'] else ""
            print(f"   {status} {r['image_name']}{desc}")
            print(f"      Tempo: {r['generation_time']}s | Elementos: {r['stats']['num_elements']} | {r['stats']['file_size_kb']:.2f} KB")
    
    print(f"\n💾 Resultados salvos em: {output_dir}")
    print(f"   - JSON: {results_json}")
    print(f"   - SVGs: {output_dir}/*.svg")
    
    if successful:
        print(f"\n💡 Abra os arquivos SVG no VS Code para comparar visualmente!")


if __name__ == "__main__":
    main()

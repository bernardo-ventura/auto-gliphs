#!/usr/bin/env python3
"""
Experimento 1: Comparação de Modelos 1B vs 8B

Este script compara a qualidade e performance dos modelos StarVector-1B e StarVector-8B
ao processar a mesma imagem.

Métricas avaliadas:
- Tempo de carregamento do modelo
- Tempo de geração do SVG
- Tamanho do arquivo SVG gerado
- Número de elementos SVG
- Qualidade visual (subjetiva - inspeção manual)

Uso:
    python exp1_model_comparison.py --image path/to/image.png
"""

import argparse
import time
import json
from pathlib import Path
from PIL import Image
import torch
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.util import process_and_rasterize_svg

def count_svg_elements(svg_text):
    """Conta o número de elementos SVG no código gerado"""
    elements = ['<path', '<line', '<rect', '<circle', '<ellipse', '<polygon', '<polyline', '<g']
    count = sum(svg_text.count(elem) for elem in elements)
    return count

def get_svg_complexity(svg_text):
    """Retorna estatísticas sobre a complexidade do SVG"""
    return {
        'num_elements': count_svg_elements(svg_text),
        'num_paths': svg_text.count('<path'),
        'num_lines': svg_text.count('<line'),
        'file_size_bytes': len(svg_text.encode('utf-8')),
        'file_size_kb': len(svg_text.encode('utf-8')) / 1024,
        'num_characters': len(svg_text),
        'num_lines_text': len(svg_text.split('\n'))
    }

def load_and_test_model(model_name, image_pil, output_dir):
    """Carrega um modelo e testa com a imagem fornecida"""
    
    results = {
        'model_name': model_name,
        'load_time_seconds': 0,
        'generation_time_seconds': 0,
        'svg_complexity': {},
        'output_files': {}
    }
    
    print(f"\n{'='*60}")
    print(f"Testando modelo: {model_name}")
    print(f"{'='*60}")
    
    # Carregar modelo
    print("📦 Carregando modelo...")
    start_time = time.time()
    model = StarVectorForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32,
        attn_implementation="eager"  # Desabilitar flash attention
    )
    model.eval()
    try:
        model.cuda()
        print("✅ Usando GPU")
    except:
        print("⚠️  Usando CPU")
    load_time = time.time() - start_time
    results['load_time_seconds'] = round(load_time, 2)
    print(f"⏱️  Tempo de carregamento: {load_time:.2f}s")
    
    # Processar imagem
    print("\n🔄 Processando imagem...")
    image = model.process_images([image_pil])[0]
    try:
        image = image.to(torch.float16).cuda()
    except:
        pass
    
    batch = {"image": image}
    
    # Gerar SVG
    print("🎨 Gerando SVG...")
    start_time = time.time()
    raw_svg = model.generate_im2svg(
        batch, 
        max_length=4000,
        temperature=1.5,
        length_penalty=-1,
        repetition_penalty=3.1
    )[0]
    generation_time = time.time() - start_time
    results['generation_time_seconds'] = round(generation_time, 2)
    print(f"⏱️  Tempo de geração: {generation_time:.2f}s")
    
    # Processar SVG
    print("✨ Processando SVG...")
    svg, raster_image = process_and_rasterize_svg(raw_svg)
    
    # Analisar complexidade
    complexity = get_svg_complexity(svg)
    results['svg_complexity'] = complexity
    
    print(f"\n📊 Estatísticas do SVG:")
    print(f"   - Elementos totais: {complexity['num_elements']}")
    print(f"   - Paths: {complexity['num_paths']}")
    print(f"   - Lines: {complexity['num_lines']}")
    print(f"   - Tamanho: {complexity['file_size_kb']:.2f} KB")
    print(f"   - Linhas de código: {complexity['num_lines_text']}")
    
    # Salvar resultados
    model_short = model_name.split('/')[-1]
    output_svg = output_dir / f"{model_short}.svg"
    output_png = output_dir / f"{model_short}_rendered.png"
    
    with open(output_svg, "w") as f:
        f.write(svg)
    
    if raster_image:
        raster_image.save(output_png)
    
    results['output_files'] = {
        'svg': str(output_svg),
        'png': str(output_png)
    }
    
    print(f"\n💾 Arquivos salvos:")
    print(f"   - SVG: {output_svg}")
    print(f"   - PNG: {output_png}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Comparar modelos StarVector 1B vs 8B')
    parser.add_argument('--image', type=str, required=True, help='Caminho para a imagem de entrada')
    parser.add_argument('--output-dir', type=str, default='experiments/results/exp1_model_comparison',
                       help='Diretório para salvar os resultados')
    args = parser.parse_args()
    
    # Criar diretório de output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Carregar imagem
    print(f"📷 Carregando imagem: {args.image}")
    image_pil = Image.open(args.image).convert('RGB')
    print(f"   Dimensões: {image_pil.size}")
    
    # Copiar imagem original para o diretório de resultados
    import shutil
    shutil.copy(args.image, output_dir / "input_image.png")
    
    # Testar ambos os modelos
    models = [
        "starvector/starvector-1b-im2svg",
        "starvector/starvector-8b-im2svg"
    ]
    
    all_results = []
    
    for model_name in models:
        try:
            result = load_and_test_model(model_name, image_pil, output_dir)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Erro ao testar {model_name}: {e}")
            continue
    
    # Salvar resultados em JSON
    results_file = output_dir / "comparison_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("📊 RESUMO DA COMPARAÇÃO")
    print(f"{'='*60}")
    
    if len(all_results) >= 2:
        model_1b = all_results[0]
        model_8b = all_results[1]
        
        print(f"\n🏃 Tempo de Carregamento:")
        print(f"   1B: {model_1b['load_time_seconds']}s")
        print(f"   8B: {model_8b['load_time_seconds']}s")
        print(f"   Diferença: +{model_8b['load_time_seconds'] - model_1b['load_time_seconds']:.2f}s para o 8B")
        
        print(f"\n⚡ Tempo de Geração:")
        print(f"   1B: {model_1b['generation_time_seconds']}s")
        print(f"   8B: {model_8b['generation_time_seconds']}s")
        print(f"   Diferença: +{model_8b['generation_time_seconds'] - model_1b['generation_time_seconds']:.2f}s para o 8B")
        
        print(f"\n📏 Complexidade do SVG:")
        print(f"   1B: {model_1b['svg_complexity']['num_elements']} elementos, {model_1b['svg_complexity']['file_size_kb']:.2f} KB")
        print(f"   8B: {model_8b['svg_complexity']['num_elements']} elementos, {model_8b['svg_complexity']['file_size_kb']:.2f} KB")
        
        print(f"\n💾 Resultados salvos em: {results_file}")
        print(f"\n💡 Compare visualmente os arquivos:")
        print(f"   - {output_dir}/starvector-1b-im2svg_rendered.png")
        print(f"   - {output_dir}/starvector-8b-im2svg_rendered.png")

if __name__ == "__main__":
    main()

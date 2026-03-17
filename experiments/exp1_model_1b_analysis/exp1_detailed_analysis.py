"""
Experimento 1: Análise Detalhada do Modelo StarVector-1B
Análise completa das capacidades do modelo 1B com diferentes tipos de imagens e parâmetros
"""

import torch
import time
import os
import json
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from starvector.model.starvector_arch import StarVectorForCausalLM
from transformers import AutoProcessor


def count_svg_elements(svg_text):
    """Conta elementos no SVG"""
    try:
        root = ET.fromstring(svg_text)
        elements = {
            'path': len(root.findall('.//{http://www.w3.org/2000/svg}path')),
            'circle': len(root.findall('.//{http://www.w3.org/2000/svg}circle')),
            'rect': len(root.findall('.//{http://www.w3.org/2000/svg}rect')),
            'line': len(root.findall('.//{http://www.w3.org/2000/svg}line')),
            'polygon': len(root.findall('.//{http://www.w3.org/2000/svg}polygon'))
        }
        return elements, sum(elements.values())
    except:
        return {}, 0


def get_svg_complexity(svg_text):
    """Calcula métricas de complexidade do SVG"""
    return {
        'size_bytes': len(svg_text.encode('utf-8')),
        'size_kb': len(svg_text.encode('utf-8')) / 1024,
        'num_characters': len(svg_text),
        'num_lines': svg_text.count('\n') + 1
    }


def test_image_with_params(model, processor, image_path, temperature, max_length, repetition_penalty):
    """Testa uma imagem com parâmetros específicos"""
    
    # Carregar e processar imagem
    image = Image.open(image_path)
    
    # Preparar input
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Medir tempo de geração
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=repetition_penalty
        )
    
    generation_time = time.time() - start_time
    
    # Decodificar output
    svg_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Calcular métricas
    elements, total_elements = count_svg_elements(svg_text)
    complexity = get_svg_complexity(svg_text)
    
    return {
        'svg_text': svg_text,
        'generation_time': generation_time,
        'elements': elements,
        'total_elements': total_elements,
        'complexity': complexity,
        'parameters': {
            'temperature': temperature,
            'max_length': max_length,
            'repetition_penalty': repetition_penalty
        }
    }


def run_experiment():
    """Executa experimento completo"""
    
    print("=" * 80)
    print("EXPERIMENTO 1: Análise Detalhada do StarVector-1B")
    print("=" * 80)
    
    # Criar diretórios de saída
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Carregar modelo
    print("\n1. Carregando modelo StarVector-1B...")
    model_name = "starvector/starvector-1b-im2svg"
    
    load_start = time.time()
    model = StarVectorForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    load_time = time.time() - load_start
    
    print(f"✓ Modelo carregado em {load_time:.2f}s")
    
    # Imagens de teste
    test_images = [
        ("/home/beventura/UC/star-vector/assets/examples/image2svg/sample-18.png", "icon"),
    ]
    
    # Configurações de parâmetros para testar
    param_configs = [
        # Temperatura baixa (mais conservador)
        {'temperature': 0.7, 'max_length': 4000, 'repetition_penalty': 1.2},
        # Padrão
        {'temperature': 1.5, 'max_length': 4000, 'repetition_penalty': 1.0},
        # Temperatura alta (mais criativo)
        {'temperature': 2.0, 'max_length': 4000, 'repetition_penalty': 1.0},
        # Comprimento maior
        {'temperature': 1.5, 'max_length': 6000, 'repetition_penalty': 1.0},
        # Anti-repetição forte
        {'temperature': 1.5, 'max_length': 4000, 'repetition_penalty': 1.5},
    ]
    
    all_results = []
    
    # Testar cada imagem
    for img_path, img_type in test_images:
        print(f"\n2. Testando imagem: {Path(img_path).name} (tipo: {img_type})")
        
        if not os.path.exists(img_path):
            print(f"⚠ Imagem não encontrada: {img_path}")
            continue
        
        image_results = {
            'image_path': img_path,
            'image_type': img_type,
            'configurations': []
        }
        
        # Testar cada configuração de parâmetros
        for i, params in enumerate(param_configs, 1):
            print(f"\n   Configuração {i}/{len(param_configs)}: temp={params['temperature']}, "
                  f"max_len={params['max_length']}, rep_pen={params['repetition_penalty']}")
            
            try:
                result = test_image_with_params(
                    model, processor, img_path,
                    params['temperature'],
                    params['max_length'],
                    params['repetition_penalty']
                )
                
                print(f"   ✓ Tempo: {result['generation_time']:.2f}s")
                print(f"   ✓ Elementos: {result['total_elements']}")
                print(f"   ✓ Tamanho: {result['complexity']['size_kb']:.2f} KB")
                
                # Salvar SVG
                svg_filename = f"{Path(img_path).stem}_config{i}.svg"
                svg_path = output_dir / svg_filename
                with open(svg_path, 'w') as f:
                    f.write(result['svg_text'])
                
                result['svg_file'] = str(svg_filename)
                del result['svg_text']  # Não incluir no JSON (muito grande)
                
                image_results['configurations'].append(result)
                
            except Exception as e:
                print(f"   ✗ Erro: {e}")
                image_results['configurations'].append({
                    'error': str(e),
                    'parameters': params
                })
        
        all_results.append(image_results)
    
    # Salvar resultados em JSON
    json_path = output_dir / "experiment_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("EXPERIMENTO CONCLUÍDO")
    print("=" * 80)
    print(f"\nResultados salvos em: {output_dir}")
    print(f"- Arquivo JSON: {json_path}")
    print(f"- Arquivos SVG: {output_dir}/*.svg")
    
    # Resumo estatístico
    print("\nRESUMO ESTATÍSTICO:")
    
    total_tests = sum(len(img['configurations']) for img in all_results)
    successful_tests = sum(
        1 for img in all_results 
        for config in img['configurations'] 
        if 'error' not in config
    )
    
    print(f"- Total de testes: {total_tests}")
    print(f"- Testes bem-sucedidos: {successful_tests}")
    print(f"- Taxa de sucesso: {successful_tests/total_tests*100:.1f}%")
    
    # Estatísticas de tempo e tamanho
    times = [
        config['generation_time'] 
        for img in all_results 
        for config in img['configurations'] 
        if 'generation_time' in config
    ]
    
    sizes = [
        config['complexity']['size_kb'] 
        for img in all_results 
        for config in img['configurations'] 
        if 'complexity' in config
    ]
    
    if times:
        print(f"\nTEMPO DE GERAÇÃO:")
        print(f"- Média: {sum(times)/len(times):.2f}s")
        print(f"- Mínimo: {min(times):.2f}s")
        print(f"- Máximo: {max(times):.2f}s")
    
    if sizes:
        print(f"\nTAMANHO DOS SVGs:")
        print(f"- Média: {sum(sizes)/len(sizes):.2f} KB")
        print(f"- Mínimo: {min(sizes):.2f} KB")
        print(f"- Máximo: {max(sizes):.2f} KB")


if __name__ == "__main__":
    run_experiment()

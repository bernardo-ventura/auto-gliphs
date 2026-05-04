#!/usr/bin/env python3
"""
Script para testar inferência com modelo StarVector-8B pré-treinado
Usando apenas GPU 0 (GPU 1 ocupada)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Usar apenas GPU 0

from PIL import Image
import torch
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.util import process_and_rasterize_svg
import time

def main():
    print("🧪 Testando StarVector-8B - Inferência (GPU 0 apenas)")
    print("="*60)
    
    # Verificar GPUs disponíveis
    print(f"🖥️  GPU disponível: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    mem_total = props.total_memory / 1024**3
    print(f"   Memória total: {mem_total:.1f} GB")
    
    # Carregar imagem
    image_path = "assets/examples/sample-0.png"
    print(f"\n📷 Carregando imagem: {image_path}")
    image_pil = Image.open(image_path).convert('RGB')
    print(f"   Dimensões: {image_pil.size}")
    
    # Carregar modelo 8B
    model_name = "starvector/starvector-8b-im2svg"
    print(f"\n📦 Carregando modelo: {model_name}")
    print("   ⚠️  Modelo 8B (~16GB) em GPU de 12GB - pode dar OOM")
    
    start_time = time.time()
    
    try:
        model = StarVectorForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # Usar implementação eager (sem Flash Attention)
        )
        model.cuda()
        model.eval()
        
        load_time = time.time() - start_time
        print(f"✅ Modelo carregado!")
        print(f"⏱️  Tempo de carregamento: {load_time:.2f}s")
        
        # Mostrar uso de memória
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n📊 Memória GPU após carregar:")
        print(f"   {mem_allocated:.2f}GB alocados, {mem_reserved:.2f}GB reservados")
        
        # Processar imagem
        print("\n🔄 Processando imagem...")
        image = model.process_images([image_pil])[0]
        batch = {"image": image}
        
        # Gerar SVG
        print("🎨 Gerando SVG...")
        start_time = time.time()
        
        with torch.no_grad():
            raw_svg = model.generate_im2svg(
                batch,
                max_length=4000,
                temperature=1.5,
                length_penalty=-1,
                repetition_penalty=3.1
            )[0]
        
        gen_time = time.time() - start_time
        print(f"⏱️  Tempo de geração: {gen_time:.2f}s")
        
        # Processar e salvar SVG
        print("\n✨ Processando SVG...")
        svg, raster_image = process_and_rasterize_svg(raw_svg)
        
        output_svg = "output_8b_inference.svg"
        output_png = "output_8b_inference.png"
        
        with open(output_svg, "w") as f:
            f.write(svg)
        
        if raster_image:
            raster_image.save(output_png)
        
        print(f"\n✅ Sucesso!")
        print(f"   📄 SVG: {output_svg}")
        print(f"   🖼️  PNG: {output_png}")
        print(f"\n📊 Estatísticas:")
        print(f"   - Tamanho SVG: {len(svg.encode('utf-8')) / 1024:.2f} KB")
        print(f"   - Número de tags: {svg.count('<')}")
        
    except Exception as e:
        print(f"\n❌ Erro ao carregar/executar modelo:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

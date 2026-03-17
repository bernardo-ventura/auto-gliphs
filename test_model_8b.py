#!/usr/bin/env python3
"""
Script para testar o modelo 8B isoladamente e resolver problema do flash attention
"""

from PIL import Image
import torch
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.util import process_and_rasterize_svg
import time

def main():
    print("🧪 Testando modelo StarVector-8B isoladamente")
    print("="*60)
    
    # Carregar imagem
    image_path = "assets/examples/sample-0.png"
    print(f"📷 Carregando imagem: {image_path}")
    image_pil = Image.open(image_path).convert('RGB')
    print(f"   Dimensões: {image_pil.size}")
    
    # Carregar modelo 8B com quantização 4-bit
    model_name = "starvector/starvector-8b-im2svg"
    print(f"\n📦 Carregando modelo: {model_name}")
    print("   ⚙️  Usando quantização 4-bit para economizar memória (~2GB GPU)...")
    
    start_time = time.time()
    
    # Configurar quantização 4-bit
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    # Carregar modelo quantizado em 4-bit com CPU offload
    model = StarVectorForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    print("✅ Modelo carregado em 4-bit")
    
    load_time = time.time() - start_time
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
    gen_time = time.time() - start_time
    print(f"⏱️  Tempo de geração: {gen_time:.2f}s")
    
    # Processar SVG
    print("✨ Processando SVG...")
    svg, raster_image = process_and_rasterize_svg(raw_svg)
    
    # Salvar
    output_svg = "test_8b_output.svg"
    output_png = "test_8b_output_rendered.png"
    
    with open(output_svg, "w") as f:
        f.write(svg)
    
    if raster_image:
        raster_image.save(output_png)
    
    print(f"\n✅ Sucesso!")
    print(f"   📄 SVG: {output_svg}")
    print(f"   🖼️  PNG: {output_png}")
    print(f"\n📊 Estatísticas:")
    print(f"   - Tamanho SVG: {len(svg.encode('utf-8')) / 1024:.2f} KB")
    print(f"   - Elementos: {svg.count('<')}")

if __name__ == "__main__":
    main()

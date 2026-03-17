#!/usr/bin/env python3
"""
Script simples para testar Image2SVG com StarVector
"""
from PIL import Image
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.util import process_and_rasterize_svg

def main():
    print("🚀 Carregando modelo StarVector-1B para Image2SVG...")
    print("⚠️  Isso pode demorar alguns minutos na primeira vez (download do modelo)")
    
    # Usar o modelo menor (1B) - mais rápido para testar
    model_name = "starvector/starvector-1b-im2svg"
    
    # Carregar modelo com torch_dtype especificado
    import torch
    starvector = StarVectorForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    starvector.eval()
    
    # Se tiver CUDA disponível, use GPU
    try:
        starvector.cuda()
        print("✅ Usando GPU")
    except:
        print("⚠️  GPU não disponível, usando CPU (será mais lento)")
    
    # Carregar imagem de exemplo
    image_path = 'assets/examples/sample-18.png'  # Usar sample-18 como no quickstart
    print(f"\n📷 Carregando imagem: {image_path}")
    image_pil = Image.open(image_path)
    image_pil = image_pil.convert('RGB')  # Garantir que está em RGB
    
    # Processar imagem
    print("🔄 Processando imagem...")
    image = starvector.process_images([image_pil])[0]
    
    # Mover para GPU se disponível e converter para float16
    try:
        image = image.to(torch.float16).cuda()
    except:
        pass
    
    batch = {"image": image}
    
    # Gerar SVG com parâmetros mais adequados
    print("🎨 Gerando código SVG (isso pode demorar alguns minutos)...")
    raw_svg = starvector.generate_im2svg(
        batch, 
        max_length=4000,  # Aumentar para permitir SVGs mais complexos
        temperature=1.5,
        length_penalty=-1,
        repetition_penalty=3.1
    )[0]
    
    # Processar e renderizar SVG
    print("✨ Processando SVG gerado...")
    svg, raster_image = process_and_rasterize_svg(raw_svg)
    
    # Salvar resultados
    output_svg = "output.svg"
    output_png = "output_rendered.png"
    
    with open(output_svg, "w") as f:
        f.write(svg)
    
    if raster_image:
        raster_image.save(output_png)
        print(f"\n✅ Sucesso!")
        print(f"   📄 SVG salvo em: {output_svg}")
        print(f"   🖼️  Renderização salva em: {output_png}")
    else:
        print(f"\n✅ SVG gerado e salvo em: {output_svg}")
    
    # Mostrar preview do código SVG
    print(f"\n📝 Preview do código SVG (primeiros 500 caracteres):")
    print("-" * 60)
    print(svg[:500])
    print("-" * 60)

if __name__ == "__main__":
    main()

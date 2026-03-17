#!/usr/bin/env python3
"""
Script simples para testar Text2SVG com StarVector
"""
from PIL import Image
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.util import process_and_rasterize_svg
import torch

def main():
    print("🚀 Carregando modelo StarVector-1B para Text2SVG...")
    print("⚠️  Isso pode demorar alguns minutos se for o primeiro download")
    
    # Usar modelo de text2svg
    model_name = "starvector/starvector-1b-text2svg"
    
    # Carregar modelo com torch_dtype especificado
    starvector = StarVectorForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    starvector.eval()
    
    # Se tiver CUDA disponível, use GPU
    try:
        starvector.cuda()
        print("✅ Usando GPU")
    except:
        print("⚠️  GPU não disponível, usando CPU (será mais lento)")
    
    # Texto de entrada (prompt)
    text_prompt = "A simple house with a red roof"
    print(f"\n📝 Texto do prompt: '{text_prompt}'")
    
    # Criar uma imagem dummy (requerida pelo modelo mas não usada no text2svg)
    # Criar imagem branca 224x224
    dummy_image = Image.new('RGB', (224, 224), color='white')
    
    # Processar
    print("🔄 Processando entrada...")
    image = starvector.process_images([dummy_image])[0]
    
    # Mover para GPU se disponível
    try:
        image = image.to(torch.float16).cuda()
    except:
        pass
    
    batch = {
        "image": image,
        "caption": [text_prompt]  # Lista de captions
    }
    
    # Gerar SVG
    print("🎨 Gerando código SVG a partir do texto (isso pode demorar alguns minutos)...")
    raw_svg = starvector.model.generate_text2svg(
        batch, 
        max_length=4000,
        temperature=1.5,
        length_penalty=-1,
        repetition_penalty=3.1
    )[0]
    
    # Decodificar tokens para texto SVG
    print("✨ Decodificando SVG gerado...")
    svg_text = starvector.model.svg_transformer.tokenizer.decode(raw_svg, skip_special_tokens=True)
    
    # Processar e renderizar SVG
    svg, raster_image = process_and_rasterize_svg(svg_text)
    
    # Salvar resultados
    output_svg = "output_text2svg.svg"
    output_png = "output_text2svg_rendered.png"
    
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
    
    print(f"\n💡 Dica: Você pode editar o texto no script para gerar outros SVGs!")
    print(f"   Exemplos de prompts:")
    print(f"   - 'A red circle'")
    print(f"   - 'A tree with green leaves'")
    print(f"   - 'A star shape'")

if __name__ == "__main__":
    main()

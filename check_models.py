# check_models.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Autenticamos con tu API Key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("Consultando a Google los modelos disponibles para tu API Key...\n")

try:
    available_models = []
    for m in genai.list_models():
        # Filtramos solo los modelos que soportan generación de texto/contenido
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            available_models.append(m.name)
            
    if not available_models:
        print("\n❌ Tu API Key no tiene acceso a ningún modelo generativo. Esto confirma que necesitas habilitar la facturación en Google AI Studio/Google Cloud.")
except Exception as e:
    print(f"\n❌ Error de autenticación o permisos: {e}")
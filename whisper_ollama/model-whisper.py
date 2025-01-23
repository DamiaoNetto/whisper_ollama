import whisper
from langchain_ollama.llms import OllamaLLM
from langchain import PromptTemplate

# Carrega o modelo Whisper
model = whisper.load_model("base")

# Transcreve o áudio
result = model.transcribe("C:/Users/Netto/Desktop/Nova pasta/TOPICOS_AVANCADOS/ia.mp3", language="pt")

#  Pega texto transcrito
texto_transcrito = result['text']
print("Texto transcrito:")
print(texto_transcrito)

model_ollama = OllamaLLM(model="llama3.2:latest")  # Certifique-se de que este modelo está disponível localmente

# Define o template para o prompt de sumarização
template = """
Resuma o seguinte texto transcrito do áudio:

Texto: {texto}
Resumo:
"""

prompt = PromptTemplate(input_variables=["texto"], template=template)

# Gera o texto do prompt
prompt_text = prompt.format(texto=texto_transcrito)

print("\nPrompt enviado para o Ollama:")
print(prompt_text)

# Envia o prompt para o modelo Ollama e obtém a resposta
resumo = model_ollama.invoke(prompt_text)

# Exibe o resumo 
print("\nResumo do texto transcrito:")
print(resumo)
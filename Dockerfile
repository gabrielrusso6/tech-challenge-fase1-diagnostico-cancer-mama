# Usar imagem Python leve e estável
FROM python:3.11-slim

# Definir diretório de trabalho
WORKDIR /app

# Copiar requirements.txt primeiro para cache do Docker
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante do projeto
COPY . .

# Comando padrão para executar o pipeline principal
CMD ["python", "run_full_pipeline.py"]

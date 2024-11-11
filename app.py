from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sqlalchemy import create_engine
import pandas as pd
import psycopg2
import torch
from torch_geometric.nn import GCNConv  # Asegúrate de que tienes el tipo correcto de capa
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


# Inicializar la aplicación FastAPI
app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:4200",
    "https://p20241006cultureapp.netlify.app/",
    "https://p20241006cultureapp.netlify.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Credenciales y datos de conexión
user = 'azure_admin'
password = 'p20241006$'
host = 'p20241006.postgres.database.azure.com'  # Host de Azure
port = '5432'  # Puerto por defecto para PostgreSQL
database = 'USR_CULTURE_APP'

# Crear la URL de conexión
connection_string = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'
engine = create_engine(connection_string)

# Modelo de entrada para las recomendaciones
class Recommendations(BaseModel):
    recommended_ids: List[int]

class UserRecommendation(BaseModel):
    user_id: int




@app.get("/preferencias/")
async def get_preferencias():
    # Crear una consulta SQL para obtener todas las filas de la tabla preferencias
    query = "SELECT * FROM preferencia"
    
    # Ejecutar la consulta y cargar los resultados en un DataFrame de pandas
    df_preferencias = pd.read_sql(query, engine)
    
    # Convertir el DataFrame a una lista de diccionarios
    preferencias = df_preferencias.to_dict(orient='records')
    
    return {"preferencias": preferencias}

@app.post("/recommendations/")
async def get_recommendations(recommendations: Recommendations):
    # Convertir la lista de IDs a una cadena separada por comas
    ids_str = ','.join(map(str, recommendations.recommended_ids))

    # Crear una consulta SQL para obtener los eventos recomendados
    query = f"""
    SELECT * 
    FROM evento
    WHERE id IN ({ids_str})
    """

    # Ejecutar la consulta y cargar los resultados en un DataFrame de pandas
    df_eventos_recomendados = pd.read_sql(query, engine)

    # Convertir el DataFrame a una lista de diccionarios
    eventos_recomendados = df_eventos_recomendados.to_dict(orient='records')

    return {"recommended_events": eventos_recomendados}

@app.post("/recommend/")
async def recommend(user: UserRecommendation): 
    user_id = user_encoder.transform([user.user_id])[0]
    # Obtener índices de productos con los que el usuario no interactuó aún
    uninteracted_items = torch.tensor([i + num_usuarios for i in range(num_items) if i not in df_preferencias[df_preferencias['user_id'] == user_id]['event_id'].values])

    # Crear edge_index para user_data
    user_edge_index = torch.tensor([[user_id] * len(uninteracted_items), uninteracted_items], dtype=torch.long)

    # Crear data object de productos no interactuados
    user_data = Data(edge_index=user_edge_index, x=node_features)
   # Realizar la predicción
    with torch.no_grad():
        predictions = model(user_data)

    # Ordenar predicciones de forma descendente
    _, sorted_indices = torch.sort(predictions, descending=True)

    # Obtener Top 10 recomendaciones
    recommended_items = uninteracted_items[sorted_indices[:10]]
    recommended_item_ids = recommended_items.tolist()
    

    return {"recommended_items": recommended_item_ids}

@app.post("/recommendById/")
async def recommend(user: UserRecommendation): 

    user_id = user.user_id
        # Obtener índices de productos con los que el usuario no interactuó aún
    uninteracted_items = torch.tensor([i + num_usuarios for i in range(num_items) if i not in df_preferencias[df_preferencias['user_id'] == user_id]['event_id'].values])

    # Crear edge_index para user_data
    user_edge_index = torch.tensor([[user_id] * len(uninteracted_items), uninteracted_items], dtype=torch.long)

    # Crear data object de productos no interactuados
    user_data = Data(edge_index=user_edge_index, x=node_features)
   # Realizar la predicción
    with torch.no_grad():
        predictions = model(user_data)

    # Ordenar predicciones de forma descendente
    _, sorted_indices = torch.sort(predictions, descending=True)

    # Obtener Top 10 recomendaciones
    recommended_items = uninteracted_items[sorted_indices[:10]]
    recommended_item_ids = recommended_items.tolist()
    

    ids_str = ','.join(map(str, recommended_item_ids))

    # Crear una consulta SQL para obtener los eventos recomendados
    query = f"""
    SELECT * 
    FROM evento
    WHERE id IN ({ids_str})
    """

    # Ejecutar la consulta y cargar los resultados en un DataFrame de pandas
    df_eventos_recomendados = pd.read_sql(query, engine)

    # Convertir el DataFrame a una lista de diccionarios
    eventos_recomendados = df_eventos_recomendados.to_dict(orient='records')

    return {"recommended_events": eventos_recomendados}

@app.get("/recommendById/{user_id}")
async def recommend(user_id: int): 
    #Modelo 
    class GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.fc = torch.nn.Linear(hidden_channels * 2, out_channels)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)

            # Se aplica una capa lineal a las características de cada nodo
            edge_pred = self.fc(torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1))
            return edge_pred.squeeze()

    query_preferencias = 'SELECT * FROM preferencia'  
    df_preferencias = pd.read_sql(query_preferencias, engine)

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df_preferencias['user_id'] = user_encoder.fit_transform(df_preferencias['user_id'])
    df_preferencias['event_id'] = item_encoder.fit_transform(df_preferencias['event_id'])

        #Creamos grafo
        # Crear vértices de las interacciones usuario-producto
    edge_index = torch.tensor(
        np.array([df_preferencias['user_id'].values, df_preferencias['event_id'].values]), 
        dtype=torch.long
    )

        # Crear atributos de los vértices (ratings)
    edge_attr = torch.tensor(df_preferencias['rating'].values, dtype=torch.float)

        # Crear PyTorch Geometric data object
    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    num_usuarios = df_preferencias['user_id'].nunique()
    num_items = df_preferencias['event_id'].nunique()
    num_nodos = num_usuarios + num_items
        # Crear características de los nodos
    node_features = torch.eye(num_nodos)
        # Agregar las características de los nodos
    data.x = node_features
        # Inicializar modelo
    model = GCN(in_channels=node_features.size(1), hidden_channels=16, out_channels=1)


    if user_id not in user_encoder.classes_:
        return {"error": f"User ID {user_id} not found in encoder classes."}
    # Transformar el user_id a su correspondiente índice interno
    user_id_transformed = user_encoder.transform([user_id])[0]
    # Obtener índices de eventos con los que el usuario no interactuó aún
    uninteracted_items = torch.tensor([i + num_usuarios for i in range(num_items) if i not in df_preferencias[df_preferencias['user_id'] == user_id]['event_id'].values])

    #en caso no haya eventos no interactuados
    if uninteracted_items.numel() == 0:
        # Obtener 10 eventos aleatorios en caso de no haber eventos no interactuados
        query_random_events = "SELECT id FROM evento ORDER BY RANDOM() LIMIT 10"
        df_random_events = pd.read_sql(query_random_events, engine)
        random_event_ids = df_random_events['id'].tolist()
        ids_str = ','.join(map(str, random_event_ids))

        query = f"""
        SELECT e.*, f.isfavorite, p.rating
        FROM evento e
        LEFT JOIN favorito f ON e.id = f.evento_id
        LEFT JOIN preferencia p ON e.id = p.event_id
        WHERE e.id IN ({ids_str}) and e.archivado = false
        """

        df_eventos_recomendados = pd.read_sql(query, engine)
        df_eventos_recomendados = df_eventos_recomendados.replace({pd.NA: None, np.nan: None})  # Reemplazar NaN con None
        eventos_recomendados = df_eventos_recomendados.to_dict(orient='records')

        return {"recommended_events": eventos_recomendados}


    # Crear edge_index para user_data
    user_edge_index = torch.tensor([[user_id] * len(uninteracted_items), uninteracted_items], dtype=torch.long)

    # Crear data object de eventos no interactuados
    user_data = Data(edge_index=user_edge_index, x=node_features)
   # Realizar la predicción
    with torch.no_grad():
        predictions = model(user_data)

    # Ordenar predicciones de forma descendente
    _, sorted_indices = torch.sort(predictions, descending=True)

    # Obtener Top 10 recomendaciones
    recommended_items = uninteracted_items[sorted_indices[:10]]
    recommended_item_ids = recommended_items.tolist()
    
   # Manejo de resultados vacíos
    if len(recommended_item_ids) == 0:
        # Obtener 10 eventos aleatorios en caso de no haber recomendaciones
        query_random_events = "SELECT id FROM evento ORDER BY RANDOM() LIMIT 10"
        df_random_events = pd.read_sql(query_random_events, engine)
        random_event_ids = df_random_events['id'].tolist()
        ids_str = ','.join(map(str, random_event_ids))

        query = f"""
        SELECT e.*, f.isfavorite, p.rating
        FROM evento e
        LEFT JOIN favorito f ON e.id = f.evento_id
        LEFT JOIN preferencia p ON e.id = p.event_id
        WHERE e.id IN ({ids_str}) and e.archivado = false
        """

        df_eventos_recomendados = pd.read_sql(query, engine)
        df_eventos_recomendados = df_eventos_recomendados.replace({pd.NA: None, np.nan: None})  # Reemplazar NaN con None
        eventos_recomendados = df_eventos_recomendados.to_dict(orient='records')

        return {"recommended_events": eventos_recomendados}    

    ids_str = ','.join(map(str, recommended_item_ids))

    # Crear una consulta SQL para obtener los eventos recomendados
    query = f"""
    SELECT e.*, f.isfavorite, p.rating
    FROM evento e
    LEFT JOIN favorito f ON e.id = f.evento_id
    LEFT JOIN preferencia p ON e.id = p.event_id
    WHERE e.id IN ({ids_str}) and e.archivado = false
    """

    # Ejecutar la consulta y cargar los resultados en un DataFrame de pandas
    df_eventos_recomendados = pd.read_sql(query, engine)
    df_eventos_recomendados = df_eventos_recomendados.replace({pd.NA: None, np.nan: None})  # Reemplazar NaN con None
    # Convertir el DataFrame a una lista de diccionarios
    eventos_recomendados = df_eventos_recomendados.to_dict(orient='records')

    return {"recommended_events": eventos_recomendados}
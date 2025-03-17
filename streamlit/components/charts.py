import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Optional

def plot_density_distribution(
    data: pd.DataFrame,
    metric: str = 'commercial_density',
    title: Optional[str] = None
) -> go.Figure:
    """
    Crea un histograma de distribución de densidad.
    
    Args:
        data: DataFrame con datos
        metric: Columna a visualizar
        title: Título opcional
        
    Returns:
        Figura de Plotly
    """
    fig = px.histogram(
        data,
        x=metric,
        nbins=50,
        title=title or f"Distribución de {metric}",
        labels={metric: metric.replace('_', ' ').title()}
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title=metric.replace('_', ' ').title(),
        yaxis_title="Frecuencia"
    )
    
    return fig

def plot_business_types(
    data: pd.DataFrame,
    top_n: int = 10
) -> go.Figure:
    """
    Crea un gráfico de barras de tipos de negocios.
    
    Args:
        data: DataFrame con datos
        top_n: Número de tipos a mostrar
        
    Returns:
        Figura de Plotly
    """
    # Procesar tipos de negocios
    business_counts = pd.Series(
        [bt for bts in data['business_types'] for bt in eval(bts)]
    ).value_counts().head(top_n)
    
    fig = px.bar(
        business_counts,
        title=f"Top {top_n} Tipos de Negocios",
        labels={'index': 'Tipo de Negocio', 'value': 'Cantidad'}
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    return fig

def plot_metrics_evolution(
    data: pd.DataFrame,
    metrics: List[str]
) -> go.Figure:
    """
    Crea un gráfico de líneas para evolución de métricas.
    
    Args:
        data: DataFrame con datos
        metrics: Lista de métricas a visualizar
        
    Returns:
        Figura de Plotly
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[metric],
                name=metric.replace('_', ' ').title(),
                mode='lines'
            ),
            secondary_y=(i == 1)
        )
    
    fig.update_layout(
        title="Evolución de Métricas",
        xaxis_title="Fecha",
        hovermode="x unified"
    )
    
    return fig

def plot_cluster_characteristics(
    data: pd.DataFrame,
    features: List[str]
) -> go.Figure:
    """
    Crea un gráfico de radar para características de clusters.
    
    Args:
        data: DataFrame con datos
        features: Lista de features a visualizar
        
    Returns:
        Figura de Plotly
    """
    fig = go.Figure()
    
    for cluster in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster]
        
        fig.add_trace(go.Scatterpolar(
            r=[cluster_data[f].mean() for f in features],
            theta=features,
            fill='toself',
            name=f'Cluster {cluster}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Características de Clusters"
    )
    
    return fig

def plot_prediction_comparison(
    data: pd.DataFrame
) -> go.Figure:
    """
    Crea un scatter plot de predicciones vs valores reales.
    
    Args:
        data: DataFrame con predicciones
        
    Returns:
        Figura de Plotly
    """
    fig = px.scatter(
        data,
        x='commercial_density',
        y='prediction',
        title="Predicción vs Real",
        labels={
            'commercial_density': 'Densidad Comercial Real',
            'prediction': 'Predicción'
        }
    )
    
    # Agregar línea de 45 grados
    fig.add_trace(
        go.Scatter(
            x=[data['commercial_density'].min(), data['commercial_density'].max()],
            y=[data['commercial_density'].min(), data['commercial_density'].max()],
            mode='lines',
            name='Predicción Perfecta',
            line=dict(dash='dash', color='red')
        )
    )
    
    return fig
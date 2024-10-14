import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Set the seaborn theme for better aesthetics
sns.set()

def report(preds):

    # Example DataFrame (Replace this with your actual data)
    df = preds.reset_index()
    
    # Plotting the time series
    plt.figure(figsize=(12, 7))
    plt.plot(df['index'], df['Q - OBS'], label='Valor Real', color='black', linestyle='-', linewidth=2)
    plt.plot(df['index'], df['Q - BASE'], label='Previsão Base', linestyle=':')
    plt.plot(df['index'], df['Q - ONS'], label='Previsão ONS', linestyle='-.')
    plt.plot(df['index'], df['Q - OPT'], label='Previsão Calibrada', linestyle='--')
    
    # Setting titles and labels in Portuguese
    plt.title('Comparação das Séries ao Longo do Tempo')
    plt.xlabel('Data')
    plt.ylabel('Vazão de Água (m3/dia)')
    plt.legend()
    
    # Show the plot
    plt.show()
    
    # Acumular os valores ao longo do tempo
    df['Q - OBS Acumulado'] = df['Q - OBS'].cumsum()
    df['Q - BASE Acumulado'] = df['Q - BASE'].cumsum()
    df['Q - OPT Acumulado'] = df['Q - OPT'].cumsum()
    df['Q - ONS Acumulado'] = df['Q - ONS'].cumsum()
    
    # Plotting the accumulated time series
    plt.figure(figsize=(12, 7))
    plt.plot(df['index'], df['Q - OBS Acumulado'], label='Valor Real Acumulado', color='black', linestyle='-', linewidth=2)
    plt.plot(df['index'], df['Q - BASE Acumulado'], label='Previsão Base Acumulada', linestyle=':')
    plt.plot(df['index'], df['Q - OPT Acumulado'], label='Previsão Calibrada Acumulada', linestyle='--')
    plt.plot(df['index'], df['Q - ONS Acumulado'], label='Previsão ONS Acumulada', linestyle='-.')
    
    # Setting titles and labels in Portuguese
    plt.title('Comparação das Séries Acumuladas ao Longo do Tempo')
    plt.xlabel('Data')
    plt.ylabel('Vazão de Água Acumulada (m3)')
    plt.legend()
    
    # Show the plot
    plt.show()
    
    # Plotting the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Q - OBS'], df['Q - BASE'], label='Previsão Base', alpha=0.6)
    plt.scatter(df['Q - OBS'], df['Q - OPT'], label='Previsão Calibrada', alpha=0.6)
    plt.scatter(df['Q - OBS'], df['Q - ONS'], label='Previsão ONS', alpha=0.6)
    
    # Adding a diagonal line (perfect match line)
    plt.plot([df['Q - OBS'].min(), df['Q - OBS'].max()],
             [df['Q - OBS'].min(), df['Q - OBS'].max()],
             color='black', linestyle='--', label='Linha de Perfeição')
    
    # Setting titles and labels in Portuguese
    plt.title('Gráfico de Dispersão: Vazão Gerada vs Vazão Observada')
    plt.xlabel('Vazão Observada (m3/dia)')
    plt.ylabel('Vazão Gerada (m3/dia)')
    plt.legend()
    
    # Show the plot
    plt.show()
    
    # Calculate residuals
    df['Resíduo - BASE'] = df['Q - BASE'] - df['Q - OBS']
    df['Resíduo - OPT'] = df['Q - OPT'] - df['Q - OBS']
    df['Resíduo - ONS'] = df['Q - ONS'] - df['Q - OBS']
    
    # Plotting the residuals
    plt.figure(figsize=(12, 7))
    plt.plot(df['index'], df['Resíduo - BASE'], label='Resíduo Base', linestyle=':')
    plt.plot(df['index'], df['Resíduo - OPT'], label='Resíduo Calibrada', linestyle='--')
    plt.plot(df['index'], df['Resíduo - ONS'], label='Resíduo ONS', linestyle='-.')
    
    # Adding a horizontal line at zero to indicate perfect prediction
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    
    # Setting titles and labels in Portuguese
    plt.title('Gráfico de Resíduos')
    plt.xlabel('Data')
    plt.ylabel('Resíduo (m3/dia)')
    plt.legend()
    
    # Show the plot
    plt.show()

    # Function to calculate the permanence curve
    def calculate_permanence_curve(series):
        sorted_series = np.sort(series)[::-1]  # Sort in descending order
        exceedance_probability = np.arange(1., len(sorted_series) + 1) / len(sorted_series) * 100
        return exceedance_probability, sorted_series
    
    # Calculate permanence curves
    exceedance_obs, sorted_obs = calculate_permanence_curve(df['Q - OBS'])
    exceedance_base, sorted_base = calculate_permanence_curve(df['Q - BASE'])
    exceedance_opt, sorted_opt = calculate_permanence_curve(df['Q - OPT'])
    exceedance_ons, sorted_ons = calculate_permanence_curve(df['Q - ONS'])
    
    # Plotting the permanence curves
    plt.figure(figsize=(12, 7))
    plt.plot(exceedance_obs, sorted_obs, label='Vazão Observada', color='black', linestyle='-', linewidth=2)
    plt.plot(exceedance_base, sorted_base, label='Vazão Gerada (Base)', linestyle=':')
    plt.plot(exceedance_opt, sorted_opt, label='Vazão Gerada (Calibrada)', linestyle='--')
    plt.plot(exceedance_ons, sorted_ons, label='Vazão Gerada (ONS)', linestyle='-.')
    
    # Setting titles and labels in Portuguese
    plt.title('Curva de Permanência das Vazões')
    plt.xlabel('Probabilidade de Excedência (%)')
    plt.ylabel('Vazão (m3/dia)')
    plt.legend()
    
    # Show the plot
    plt.show()


def interactive_report(preds, show=True):

    df = preds.reset_index()

    figures = []
    # figures = make_subplots(rows=5, cols=1)
    
    # Create the figure
    fig = go.Figure()
    
    # Add the 'Valor Real' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - OBS'], 
                             mode='lines', name='Valor Real', 
                             line=dict(color='black', width=2)))
    
    # Add the 'Previsão Base' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - BASE'], 
                             mode='lines', name='Previsão Base', 
                             line=dict(dash='dot', width=2)))
    
    # Add the 'Previsão ONS' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - ONS'], 
                             mode='lines', name='Previsão ONS', 
                             line=dict(dash='dashdot', width=2)))
    
    # Add the 'Previsão Calibrada' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - OPT'], 
                             mode='lines', name='Previsão Calibrada', 
                             line=dict(dash='dash', width=2)))
    
    # Set the title and labels
    fig.update_layout(
        title='Comparação das Séries ao Longo do Tempo',
        xaxis_title='Data',
        yaxis_title='Vazão de Água (m3/dia)',
        legend_title='Séries',
        template='plotly_white'
    )

    # Append figure
    figures.append(fig)
    # figures.append_trace(fig, row=1, col=1)
    
    # Show the interactive plot
    if show:
        fig.show()

    # Calculate residuals
    df['Resíduo - BASE'] = df['Q - BASE'] - df['Q - OBS']
    df['Resíduo - OPT'] = df['Q - OPT'] - df['Q - OBS']
    df['Resíduo - ONS'] = df['Q - ONS'] - df['Q - OBS']
    
    # Create the figure
    fig = go.Figure()
    
    # Add the diagonal 'Linha de Perfeição'
    dates = pd.date_range(start=df['index'].min(), end=df['index'].max(), periods=len(df))
    fig.add_trace(go.Scatter(x=[i for i in dates], 
                             y=[0 for i in range(len(df))], 
                             mode='lines', name='Linha de Perfeição', 
                             line=dict(color='black', width=2)))

    # # Add a horizontal line at zero (Perfect prediction)
    # fig.add_hline(y=0, line=dict(color='black', width=1), name='Linha de Perfeição')
    
    # Add the 'Resíduo Base' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Resíduo - BASE'], 
                             mode='lines', name='Resíduo Base', 
                             line=dict(dash='dot')))

    # Add the 'Resíduo ONS' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Resíduo - ONS'], 
                             mode='lines', name='Resíduo ONS', 
                             line=dict(dash='dashdot')))
    
    # Add the 'Resíduo Calibrada' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Resíduo - OPT'], 
                             mode='lines', name='Resíduo Calibrada', 
                             line=dict(dash='dash')))
        
    # Set the title and labels
    fig.update_layout(
        title='Gráfico de Resíduos',
        xaxis_title='Data',
        yaxis_title='Resíduo (m3/dia)',
        legend_title='Resíduos',
        template='plotly_white'
    )
    
    # Append figure
    figures.append(fig)
    # figures.append_trace(fig, row=4, col=1)

    # Show the interactive plot
    if show:
        fig.show()

    # Example DataFrame (Replace this with your actual data)
    df = preds.reset_index()
    
    # Acumular os valores ao longo do tempo
    df['Q - OBS Acumulado'] = df['Q - OBS'].cumsum()
    df['Q - BASE Acumulado'] = df['Q - BASE'].cumsum()
    df['Q - OPT Acumulado'] = df['Q - OPT'].cumsum()
    df['Q - ONS Acumulado'] = df['Q - ONS'].cumsum()
    
    # Create the figure
    fig = go.Figure()
    
    # Add the 'Valor Real Acumulado' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - OBS Acumulado'], 
                             mode='lines', name='Valor Real Acumulado', 
                             line=dict(color='black', width=2)))
    
    # Add the 'Previsão Base Acumulada' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - BASE Acumulado'], 
                             mode='lines', name='Previsão Base Acumulada', 
                             line=dict(dash='dot', width=2)))
    
    # Add the 'Previsão ONS Acumulada' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - ONS Acumulado'], 
                             mode='lines', name='Previsão ONS Acumulada', 
                             line=dict(dash='dashdot', width=2)))
    
    # Add the 'Previsão Calibrada Acumulada' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - OPT Acumulado'], 
                             mode='lines', name='Previsão Calibrada Acumulada', 
                             line=dict(dash='dash', width=2)))
    
    # Set the title and labels
    fig.update_layout(
        title='Comparação das Séries Acumuladas ao Longo do Tempo',
        xaxis_title='Data',
        yaxis_title='Vazão de Água Acumulada (m3)',
        legend_title='Séries Acumuladas',
        template='plotly_white'
    )

    # Append figure
    figures.append(fig)
    # figures.append_trace(fig, row=2, col=1)

    # Show the interactive plot
    if show:
        fig.show()

    # Create the figure
    fig = go.Figure()
    
    # Add the diagonal 'Linha de Perfeição'
    fig.add_trace(go.Scatter(x=[df['Q - OBS'].min(), df['Q - OBS'].max()], 
                             y=[df['Q - OBS'].min(), df['Q - OBS'].max()], 
                             mode='lines', name='Linha de Perfeição', 
                             line=dict(color='black', dash='dash')))

    # Add the 'Previsão Base' scatter trace
    fig.add_trace(go.Scatter(x=df['Q - OBS'], y=df['Q - BASE'], 
                             mode='markers', name='Previsão Base', 
                             opacity=0.6, marker=dict(size=8)))

    # Add the 'Previsão ONS' scatter trace
    fig.add_trace(go.Scatter(x=df['Q - OBS'], y=df['Q - ONS'], 
                             mode='markers', name='Previsão ONS', 
                             opacity=0.6, marker=dict(size=8)))
    
    # Add the 'Previsão Calibrada' scatter trace
    fig.add_trace(go.Scatter(x=df['Q - OBS'], y=df['Q - OPT'], 
                             mode='markers', name='Previsão Calibrada', 
                             opacity=0.6, marker=dict(size=8)))
    
    # Set the title and labels
    fig.update_layout(
        title='Gráfico de Dispersão: Vazão Gerada vs Vazão Observada',
        xaxis_title='Vazão Observada (m3/dia)',
        yaxis_title='Vazão Gerada (m3/dia)',
        legend_title='Vazões',
        template='plotly_white'
    )
    
    # Append figure
    figures.append(fig)
    # figures.append_trace(fig, row=3, col=1)

    # Show the interactive plot
    if show:
        fig.show()
        
    # Function to calculate the permanence curve
    def calculate_permanence_curve(series):
        sorted_series = np.sort(series)[::-1]  # Sort in descending order
        exceedance_probability = np.arange(1., len(sorted_series) + 1) / len(sorted_series) * 100
        return exceedance_probability, sorted_series
    
    # Calculate permanence curves
    exceedance_obs, sorted_obs = calculate_permanence_curve(df['Q - OBS'])
    exceedance_base, sorted_base = calculate_permanence_curve(df['Q - BASE'])
    exceedance_opt, sorted_opt = calculate_permanence_curve(df['Q - OPT'])
    exceedance_ons, sorted_ons = calculate_permanence_curve(df['Q - ONS'])
    
    # Create the figure
    fig = go.Figure()
    
    # Add the 'Vazão Observada' trace
    fig.add_trace(go.Scatter(x=exceedance_obs, y=sorted_obs, 
                             mode='lines', name='Vazão Observada', 
                             line=dict(color='black', width=2)))
    
    # Add the 'Vazão Gerada (Base)' trace
    fig.add_trace(go.Scatter(x=exceedance_base, y=sorted_base, 
                             mode='lines', name='Vazão Gerada (Base)', 
                             line=dict(dash='dot', width=2)))
    
    # Add the 'Vazão Gerada (ONS)' trace
    fig.add_trace(go.Scatter(x=exceedance_ons, y=sorted_ons, 
                             mode='lines', name='Vazão Gerada (ONS)', 
                             line=dict(dash='dashdot', width=2)))
    
    # Add the 'Vazão Gerada (Calibrada)' trace
    fig.add_trace(go.Scatter(x=exceedance_opt, y=sorted_opt, 
                             mode='lines', name='Vazão Gerada (Calibrada)', 
                             line=dict(dash='dash', width=2)))
    
    # Set the title and labels
    fig.update_layout(
        title='Curva de Permanência das Vazões',
        xaxis_title='Probabilidade de Excedência (%)',
        yaxis_title='Vazão (m3/dia)',
        legend_title='Curvas de Permanência',
        template='plotly_white'
    )
    
    # Append figure
    figures.append(fig)
    # figures.append_trace(fig, row=5, col=1)

    # Show the interactive plot
    if show:
        fig.show()

    return figures

def interactive_report(preds, show=True):

    df = preds.reset_index()

    figures = []
    # figures = make_subplots(rows=5, cols=1)
    
    # Create the figure
    fig = go.Figure()
    
    # Add the 'Valor Real' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - OBS'], 
                             mode='lines', name='Valor Real', 
                             line=dict(color='black', width=2)))
    
    # Add the 'Previsão Base' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - BASE'], 
                             mode='lines', name='Previsão Base', 
                             line=dict(dash='dot', width=2)))
    
    # Add the 'Previsão ONS' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - ONS'], 
                             mode='lines', name='Previsão ONS', 
                             line=dict(dash='dashdot', width=2)))
    
    # Add the 'Previsão Calibrada' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - OPT'], 
                             mode='lines', name='Previsão Calibrada', 
                             line=dict(dash='dash', width=2)))
    
    # Set the title and labels
    fig.update_layout(
        title='Comparação das Séries ao Longo do Tempo',
        xaxis_title='Data',
        yaxis_title='Vazão de Água (m3/dia)',
        legend_title='Séries',
        template='plotly_white'
    )

    # Append figure
    figures.append(fig)
    # figures.append_trace(fig, row=1, col=1)

    # Show the interactive plot
    if show:
        fig.show()

    # Calculate residuals
    df['Resíduo - BASE'] = df['Q - BASE'] - df['Q - OBS']
    df['Resíduo - OPT'] = df['Q - OPT'] - df['Q - OBS']
    df['Resíduo - ONS'] = df['Q - ONS'] - df['Q - OBS']
    
    # Create the figure
    fig = go.Figure()
    
    # Add the diagonal 'Linha de Perfeição'
    dates = pd.date_range(start=df['index'].min(), end=df['index'].max(), periods=len(df))
    fig.add_trace(go.Scatter(x=[i for i in dates], 
                             y=[0 for i in range(len(df))], 
                             mode='lines', name='Linha de Perfeição', 
                             line=dict(color='black', width=2)))

    # # Add a horizontal line at zero (Perfect prediction)
    # fig.add_hline(y=0, line=dict(color='black', width=1), name='Linha de Perfeição')
    
    # Add the 'Resíduo Base' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Resíduo - BASE'], 
                             mode='lines', name='Resíduo Base', 
                             line=dict(dash='dot')))

    # Add the 'Resíduo ONS' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Resíduo - ONS'], 
                             mode='lines', name='Resíduo ONS', 
                             line=dict(dash='dashdot')))
    
    # Add the 'Resíduo Calibrada' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Resíduo - OPT'], 
                             mode='lines', name='Resíduo Calibrada', 
                             line=dict(dash='dash')))
        
    # Set the title and labels
    fig.update_layout(
        title='Gráfico de Resíduos',
        xaxis_title='Data',
        yaxis_title='Resíduo (m3/dia)',
        legend_title='Resíduos',
        template='plotly_white'
    )
    
    # Append figure
    figures.append(fig)
    # figures.append_trace(fig, row=4, col=1)

    # Show the interactive plot
    if show:
        fig.show()
    
    # Acumular os valores ao longo do tempo
    df['Q - OBS Acumulado'] = df['Q - OBS'].cumsum()
    df['Q - BASE Acumulado'] = df['Q - BASE'].cumsum()
    df['Q - OPT Acumulado'] = df['Q - OPT'].cumsum()
    df['Q - ONS Acumulado'] = df['Q - ONS'].cumsum()
    
    # Create the figure
    fig = go.Figure()
    
    # Add the 'Valor Real Acumulado' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - OBS Acumulado'], 
                             mode='lines', name='Valor Real Acumulado', 
                             line=dict(color='black', width=2)))
    
    # Add the 'Previsão Base Acumulada' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - BASE Acumulado'], 
                             mode='lines', name='Previsão Base Acumulada', 
                             line=dict(dash='dot', width=2)))
    
    # Add the 'Previsão ONS Acumulada' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - ONS Acumulado'], 
                             mode='lines', name='Previsão ONS Acumulada', 
                             line=dict(dash='dashdot', width=2)))
    
    # Add the 'Previsão Calibrada Acumulada' trace
    fig.add_trace(go.Scatter(x=df['index'], y=df['Q - OPT Acumulado'], 
                             mode='lines', name='Previsão Calibrada Acumulada', 
                             line=dict(dash='dash', width=2)))
    
    # Set the title and labels
    fig.update_layout(
        title='Comparação das Séries Acumuladas ao Longo do Tempo',
        xaxis_title='Data',
        yaxis_title='Vazão de Água Acumulada (m3)',
        legend_title='Séries Acumuladas',
        template='plotly_white'
    )

    # Append figure
    figures.append(fig)
    # figures.append_trace(fig, row=2, col=1)

    # Show the interactive plot
    if show:
        fig.show()

    # Create the figure
    fig = go.Figure()
    
    # Add the diagonal 'Linha de Perfeição'
    fig.add_trace(go.Scatter(x=[df['Q - OBS'].min(), df['Q - OBS'].max()], 
                             y=[df['Q - OBS'].min(), df['Q - OBS'].max()], 
                             mode='lines', name='Linha de Perfeição', 
                             line=dict(color='black', dash='dash')))

    # Add the 'Previsão Base' scatter trace
    fig.add_trace(go.Scatter(x=df['Q - OBS'], y=df['Q - BASE'], 
                             mode='markers', name='Previsão Base', 
                             opacity=0.6, marker=dict(size=8)))

    # Add the 'Previsão ONS' scatter trace
    fig.add_trace(go.Scatter(x=df['Q - OBS'], y=df['Q - ONS'], 
                             mode='markers', name='Previsão ONS', 
                             opacity=0.6, marker=dict(size=8)))
    
    # Add the 'Previsão Calibrada' scatter trace
    fig.add_trace(go.Scatter(x=df['Q - OBS'], y=df['Q - OPT'], 
                             mode='markers', name='Previsão Calibrada', 
                             opacity=0.6, marker=dict(size=8)))
    
    # Set the title and labels
    fig.update_layout(
        title='Gráfico de Dispersão: Vazão Gerada vs Vazão Observada',
        xaxis_title='Vazão Observada (m3/dia)',
        yaxis_title='Vazão Gerada (m3/dia)',
        legend_title='Vazões',
        template='plotly_white'
    )
    
    # Append figure
    figures.append(fig)
    # figures.append_trace(fig, row=3, col=1)

    # Show the interactive plot
    if show:
        fig.show()
    
    # Function to calculate the permanence curve
    def calculate_permanence_curve(series):
        sorted_series = np.sort(series)[::-1]  # Sort in descending order
        exceedance_probability = np.arange(1., len(sorted_series) + 1) / len(sorted_series) * 100
        return exceedance_probability, sorted_series
    
    # Calculate permanence curves
    exceedance_obs, sorted_obs = calculate_permanence_curve(df['Q - OBS'])
    exceedance_base, sorted_base = calculate_permanence_curve(df['Q - BASE'])
    exceedance_opt, sorted_opt = calculate_permanence_curve(df['Q - OPT'])
    exceedance_ons, sorted_ons = calculate_permanence_curve(df['Q - ONS'])
    
    # Create the figure
    fig = go.Figure()
    
    # Add the 'Vazão Observada' trace
    fig.add_trace(go.Scatter(x=exceedance_obs, y=sorted_obs, 
                             mode='lines', name='Vazão Observada', 
                             line=dict(color='black', width=2)))
    
    # Add the 'Vazão Gerada (Base)' trace
    fig.add_trace(go.Scatter(x=exceedance_base, y=sorted_base, 
                             mode='lines', name='Vazão Gerada (Base)', 
                             line=dict(dash='dot', width=2)))
    
    # Add the 'Vazão Gerada (ONS)' trace
    fig.add_trace(go.Scatter(x=exceedance_ons, y=sorted_ons, 
                             mode='lines', name='Vazão Gerada (ONS)', 
                             line=dict(dash='dashdot', width=2)))
    
    # Add the 'Vazão Gerada (Calibrada)' trace
    fig.add_trace(go.Scatter(x=exceedance_opt, y=sorted_opt, 
                             mode='lines', name='Vazão Gerada (Calibrada)', 
                             line=dict(dash='dash', width=2)))
    
    # Set the title and labels
    fig.update_layout(
        title='Curva de Permanência das Vazões',
        xaxis_title='Probabilidade de Excedência (%)',
        yaxis_title='Vazão (m3/dia)',
        legend_title='Curvas de Permanência',
        template='plotly_white'
    )
    
    # Append figure
    figures.append(fig)

    # Show the interactive plot
    if show:
        fig.show()

    # Criando a figura
    fig = go.Figure()
    
    # Adicionando as séries de dados para o Reservatório Superficial
    fig.add_trace(go.Scatter(
        x=preds.index, y=preds['Rsup - BASE'], 
        mode='lines', name='Reservatório Superficial BASE', 
        line=dict(dash='dot', width=2)))  # Estilo pontilhado
    
    fig.add_trace(go.Scatter(
        x=preds.index, y=preds['Rsup - ONS'], 
        mode='lines', name='Reservatório Superficial ONS', 
        line=dict(dash='dashdot', width=2)))  # Estilo traço-ponto
    
    fig.add_trace(go.Scatter(
        x=preds.index, y=preds['Rsup - OPT'], 
        mode='lines', name='Reservatório Superficial OPT', 
        line=dict(dash='dash', width=2)))  # Estilo tracejado
    
    # Atualizando o layout para o gráfico do Reservatório Superficial
    fig.update_layout(
        title='Variação dos Níveis do Reservatório Superficial',
        xaxis_title='Data',
        yaxis_title='Nível do Reservatório Superficial (m³)',
        legend_title='Séries',
        template='plotly_white'
    )

    # Append figure
    figures.append(fig)

    # Show the interactive plot
    if show:
        fig.show()

    # Criando uma segunda figura para o Reservatório Subterrâneo
    fig = go.Figure()
    
    # Adicionando as séries de dados para o Reservatório Subterrâneo
    fig.add_trace(go.Scatter(
        x=preds.index, y=preds['Rsub - BASE'], 
        mode='lines', name='Reservatório Subterrâneo BASE', 
        line=dict(dash='dot', width=2)))  # Estilo pontilhado
    
    fig.add_trace(go.Scatter(
        x=preds.index, y=preds['Rsub - ONS'], 
        mode='lines', name='Reservatório Subterrâneo ONS', 
        line=dict(dash='dashdot', width=2)))  # Estilo traço-ponto
    
    fig.add_trace(go.Scatter(
        x=preds.index, y=preds['Rsub - OPT'], 
        mode='lines', name='Reservatório Subterrâneo OPT', 
        line=dict(dash='dash', width=2)))  # Estilo tracejado
    
    # Atualizando o layout para o gráfico do Reservatório Subterrâneo
    fig.update_layout(
        title='Variação dos Níveis do Reservatório Subterrâneo',
        xaxis_title='Data',
        yaxis_title='Nível do Reservatório Subterrâneo (m³)',
        legend_title='Séries',
        template='plotly_white'
    )

    # Append figure
    figures.append(fig)

    # Exibindo os gráficos
    if show:
        fig.show()
 
    # Convertendo Pr e Ep de mm/dia para m³/dia
    A_bacia = 6279 * 1e6  # Área da bacia em metros quadrados (ajuste com base nos dados reais)
    preds['Pr_m3'] = preds['Pr'] * A_bacia / 1000  # Pr convertida para m³/dia
    preds['Ep_m3'] = preds['Ep'] * A_bacia / 1000  # Ep convertida para m³/dia
    
    # Convertendo as vazões de m³/s para m³/dia
    preds['Q - OBS_m3_dia'] = preds['Q - OBS'] * 86400  # Vazão observada em m³/dia
    preds['Q - OPT_m3_dia'] = preds['Q - OPT'] * 86400  # Vazão otimizada em m³/dia
    
    # Agrupamento anual com os valores já convertidos
    df_annual = preds.resample('Y').sum()
    
    # Criando a figura do gráfico de barras
    fig = go.Figure()
    
    # Adicionando barras para Precipitação
    fig.add_trace(go.Bar(
        x=df_annual.index.year, 
        y=df_annual['Pr_m3'], 
        name='Precipitação (Pr)',
        marker=dict(color='blue')
    ))
    
    # Adicionando barras para Evapotranspiração
    fig.add_trace(go.Bar(
        x=df_annual.index.year, 
        y=df_annual['Ep_m3'], 
        name='Evapotranspiração (Ep)',
        marker=dict(color='orange')
    ))
    
    # Adicionando barras para Vazão Observada
    fig.add_trace(go.Bar(
        x=df_annual.index.year, 
        y=df_annual['Q - OBS_m3_dia'], 
        name='Vazão Observada (Q - OBS)',
        marker=dict(color='green')
    ))
    
    # Adicionando barras para Vazão Otimizada
    fig.add_trace(go.Bar(
        x=df_annual.index.year, 
        y=df_annual['Q - OPT_m3_dia'], 
        name='Vazão Otimizada (Q - OPT)',
        marker=dict(color='red')
    ))
    
    # Atualizando o layout do gráfico
    fig.update_layout(
        title='Balanço Hídrico Anual (m³/ano)',
        xaxis_title='Ano',
        yaxis_title='Valores Acumulados Anuais (m³)',
        barmode='group',  # Exibe barras agrupadas
        legend_title='Componentes do Balanço Hídrico',
        template='plotly_white'
    )
    
    # Append figure
    figures.append(fig)

    # Exibindo o gráfico interativo
    if show:
        fig.show()

    return figures

# Front-end prompt:
# ok I have done my own flask app. Now I want you to build the web page using html, css, javascript and tailwind. The application will have the date fields for start and end dates, it will have sliders for each parameter, the table with metrics comparing the base, ons and optimized  performances and display the charts returned. When any parameter changes, it will make a new request to the api and update all data. Remember that the response field "charts" is a list of strings, each being a "div" element with the chart. the address of the api is localhost port 5000

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import warnings; warnings.filterwarnings('ignore')

from modules.smap import ModeloSmapDiario

from sklearn.metrics import mean_squared_error
from modules.metrics import (
    nash_sutcliffe_efficacy,
    relative_error_coefficient,
    correlation_coefficient,
    mean_error,
    mean_abs_error,
    normalized_rmse,
    rmse
)

from modules.visu import report, interactive_report

df = pd.read_csv('data/bacia-camargos.csv')
df['Ep'] = df['Ep'].str.replace(',', '.').astype('float')
df['Pr'] = df['Pr'].str.replace(',', '.').astype('float')
df.set_index(pd.to_datetime(df['data']), inplace=True)
df.drop('data', axis=1, inplace=True)

output_base = pd.read_csv('data/optimization/output_base_full.csv', index_col=0)
output_ons = pd.read_csv('data/optimization/output_ons_full.csv', index_col=0)

output_base.set_index(pd.to_datetime(output_base.index), inplace=True)
output_ons.set_index(pd.to_datetime(output_ons.index), inplace=True)

output_base.columns += ' - BASE'
output_ons.columns += ' - ONS'



app = Flask(__name__)
CORS(app)

# Route to handle data submission and return metrics
@app.route('/run_model', methods=['POST'])
def run_model():
    args = request.json
    
    params = args['params']
    start_date = args['start_date']
    end_date = args['end_date']

    data = df[start_date: end_date]
    
    Ep = data['Ep'].tolist()
    Pr = data['Pr'].tolist()
    
    result = ModeloSmapDiario(Ep=Ep, Pr=Pr, **params)
    result = pd.DataFrame(result, index=data.index)
    
    output_opt = result.copy()
    output_opt.columns += ' - OPT'
    preds = pd.concat([output_base.loc[output_opt.index], output_ons.loc[output_opt.index], output_opt], axis=1)
    
    preds.index = data.index
    preds.index.name = 'index'
    preds['Q - OBS'] = data['Qobs']
    preds['Q - OPT'] = preds['Q - OPT']
    preds['Rsub - OPT'] = preds['Rsub - OPT']
    preds['Rsup - OPT'] = preds['Rsup - OPT']
    preds = pd.concat([preds, data], axis=1)
    
    y_true = preds['Q - OBS']
    columns = ['Q - BASE', 'Q - ONS', 'Q - OPT']
    
    stats = []
    for pred in columns:
        y_pred = preds[pred]
    
        # Calculate the score (Mean Squared Error in this case)
        mse = mean_squared_error(y_true, y_pred)
        cef = nash_sutcliffe_efficacy(y_true, y_pred)
        cer = relative_error_coefficient(y_true, y_pred)
        soma_coef = cef + cer
        
        cc = correlation_coefficient(y_true, y_pred)
        me = mean_error(y_true, y_pred)
        mae = mean_abs_error(y_true, y_pred)
        rmse_norm = normalized_rmse(y_true, y_pred)
        RMSE = rmse(y_true, y_pred)
    
        stats.append({
            'cef': cef,
            'cer': cer,
            'soma_coef': soma_coef,
            'cc': cc,
            'me': me,
            'mae': mae,
            'rmse_norm': rmse_norm,
            'rmse': RMSE}
        )
    
    stats = pd.DataFrame(stats, index=columns)
    stats = stats.to_dict(orient='records')
    
    figures = interactive_report(preds, show=False)
    figures_html = [fig.to_html(full_html=False, include_plotlyjs='cdn').replace('\n', '') for fig in figures]
    
    # Return the results as JSON
    return jsonify({
        'stats': stats,
        'charts': figures_html,
    })

if __name__ == '__main__':
    app.run(debug=True)

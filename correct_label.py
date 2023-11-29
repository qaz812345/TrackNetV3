import os
import cv2
import json
import parse
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output

from dataset import data_dir
from utils.general import *

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--host', type=str, default='127.0.0.1')
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()

split = args.split
host = args.host
debug = args.debug

# Evaluation result file list
if split == 'train':
    eval_file_list = [
        # {'label': label_name, 'value': json_path},
    ]
elif split == 'val':
    eval_file_list = [
        # {'label': label_name, 'value': json_path},
    ]
elif split == 'test':
    eval_file_list = [
        {'label': 'tracknet', 'value': 'test/tracknet_eval/test_eval_analysis_weight.json'},
        {'label': 'tracknetv3', 'value': 'test/tracknetv3_eval/test_eval_analysis_weight.json'}
    ]
else:
    raise ValueError(f'Invalid split: {split}')

# Init global variables
pred_types = ['TP', 'TN', 'FP1', 'FP2', 'FN']
pred_types_map = {pred_type: i for i, pred_type in enumerate(pred_types)}
prev_click = 0
match_id, rally_id, frame_id = None, None, None
x_gt, y_gt, vis_gt = None, None, None
x_pred, y_pred, vis_pred = None, None, None
x_correct, y_correct, vis_correct = None, None, None

# Generate drop down list values of rally id
rally_keys = []
rally_dirs = get_rally_dirs(data_dir, split)
rally_dirs = [os.path.join(data_dir, d) for d in rally_dirs]
for rally_dir in rally_dirs:
    file_format_str = os.path.join('{}', 'match', '{}', 'frame', '{}')
    _, match_id, rally_id = parse.parse(file_format_str, rally_dir)
    rally_keys.append(f'{match_id}_{rally_id}')
rally_id_map = {k: i for i, k in enumerate(rally_keys)}

# Load drop frame dict if split is test
if split == 'test':
    drop_frame_dict = json.load(open(f'{data_dir}/drop_frame.json'))
    start_f, end_f = drop_frame_dict['start'], drop_frame_dict['end']
else:
    start_f, end_f = None, None

# Create dash app
app = dash.Dash(__name__)
app.layout = html.Div(children=[
    # Drop down lists
    html.Div(children=[
        html.Div(children=[
            html.Label(['Model:'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(eval_file_list, eval_file_list[0]['value'], id='eval-file-dropdown')
        ], style=dict(width='20%', margin='10px')),
        html.Div(children=[
            html.Label(['Rally ID:'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(rally_keys, rally_keys[0], id='rally-dropdown')
        ], style={'width':'20%', 'margin':'10px'})
    ], style={'display':'flex', 'justify-content':'center', 'text-align':'center'}),
    dcc.Input(id='write-mag', type='hidden', value='not_saved'),
    dcc.Input(id='reset-mag', type='hidden', value='not_reseted'),
    # Time series plot
    html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='time_fig',
                figure=go.Figure(),
                config={'scrollZoom':True}
            ),
        ], style=dict(width='90%')),
    ], style={'display':'flex', 'justify-content':'center', 'text-align':'center'}),
    # Buttons
    html.Div(children=[
        html.Button('Write Result', id='write-btn', n_clicks=0, style={'width':'160px', 'height':'40px', 'margin': '10px'}),
        html.Button('Reset Label', id='reset-btn', n_clicks=0, style={'width':'160px', 'height':'40px', 'margin': '10px'})
    ], style={'display':'flex', 'justify-content': 'center', 'align-items': 'center'}),
    # Frame plot
    html.Div(children=[
        dcc.Graph(
            id='frame_fig',
            figure=go.Figure(),
            config={'scrollZoom':True}
        ),
    ], style={'display':'flex', 'justify-content':'center', 'align-items': 'center'}),
])


@app.callback(
    Output('time_fig', 'figure'),
    [Input('eval-file-dropdown', 'value'),
    Input('rally-dropdown', 'value')]
)
def change_dropdown(eval_file, rally_key):
    global match_id, rally_id, x_gt, y_gt, vis_gt, x_pred, y_pred, vis_pred, x_correct, y_correct, vis_correct
    
    # Bar chart settings
    bar_width = 1
    y_min, y_max = - 0.2, 1.5
    colors = {'TP': '#65AD6C', 'TN': '#D47D7D', 'FP1': 'green', 'FP2': 'red', 'FN': 'blue'}

    # Parse rally key
    rally_key_splits = rally_key.split('_')
    match_id, rally_id = rally_key_splits[0], '_'.join(rally_key_splits[1:])

    # Read ground truth label
    csv_gt = os.path.join(data_dir, split, f'match{match_id}', 'csv', f'{rally_id}_ball.csv')
    gt_df = pd.read_csv(csv_gt, encoding='utf8')
    x_gt, y_gt, vis_gt = np.array(gt_df['X']), np.array(gt_df['Y']), np.array(gt_df['Visibility'])

    # Init correct result
    x_correct, y_correct, vis_correct = np.array(gt_df['X']), np.array(gt_df['Y']), np.array(gt_df['Visibility'])

    # Read prediction results
    print(f'File: {eval_file}')
    eval_dict = json.load(open(eval_file))['pred_dict'][rally_key]
    x_pred, y_pred, vis_pred = np.array(eval_dict['X']), np.array(eval_dict['Y']), np.array(eval_dict['Visibility'])

    # Parse prediction result into stack bar chart data
    bar_list = {}
    timestamp = np.arange(len(gt_df))
    for pred_type in pred_types:
        bar_list[pred_type] = (np.array(eval_dict['Type']) == pred_types_map[pred_type]).astype('int')
    bar_list['Error'] = bar_list['FN'] + bar_list['FP1'] + bar_list['FP2']
    bar_list['TP'] = bar_list['TP'] * y_min
    bar_list['TN'] = bar_list['TN'] * y_min
    
    # Plot stack bar chart
    hover_data = np.stack([x_gt, y_gt, vis_gt, x_pred, y_pred, vis_pred], axis=1)
    time_fig = go.Figure()
    for pred_type in pred_types: 
        time_fig.add_trace(
            go.Bar(x=timestamp, y=bar_list[pred_type], customdata=hover_data,
                   width=bar_width, marker_color=colors[pred_type], name=pred_type,
                   legendgroup=pred_type, showlegend=True),
        )
    
    # Visualize effective trajectory
    if split == 'test':
        # The moment of serve
        time_fig.add_vline(x=start_f[rally_key]-bar_width/2, line_width=1, line_dash='dash', line_color='gray')
        # The moment of the ball touch ground
        time_fig.add_vline(x=end_f[rally_key]-bar_width/2, line_width=1, line_dash='dash', line_color='gray')

    time_fig.update_yaxes(title_text='Error Count', range=[y_min, y_max], fixedrange=True)
    time_fig.update_xaxes(title_text='Frame ID')
    time_fig.update_layout(barmode='stack', dragmode='pan', clickmode='event+select',
                           margin={'l':20, 'r':20, 't':50, 'b':10}, height=300,
                           title_text=f'Rally {rally_key} Error Distribution', title_x=0.5, legend_title='Error Type')
    time_fig.update_traces(
        hovertemplate="<br>".join([
            "frame id: %{x}",
            "label: ( %{customdata[0]}, %{customdata[1]} ), vis: %{customdata[2]}",
            "pred: ( %{customdata[3]}, %{customdata[4]} ), vis: %{customdata[5]}",
        ])
    )

    return time_fig


@app.callback(
    Output('write-mag', 'value'),
    Input('write-btn', 'n_clicks')
)
def save_corrected_result(n_clicks):
    global match_id, rally_id, x_correct, y_correct, vis_correct
    if n_clicks:
        correct_dir = os.path.join(data_dir, split, f'match{match_id}', 'corrected_csv')
        if not os.path.exists(correct_dir):
            os.makedirs(correct_dir)
        
        out_csv_file = os.path.join(correct_dir, f'{rally_id}_ball.csv')
        df = pd.DataFrame({'Frame': [i for i in range(len(vis_correct))],
                           'Visibility': vis_correct,
                           'X': x_correct,
                           'Y': y_correct})
        df.to_csv(out_csv_file, index=False)
        print(f'{out_csv_file} saved')

        return f'{out_csv_file}_saved'


@app.callback(
    Output('frame_fig', 'figure'),
    [Input('time_fig', 'hoverData'),
    Input('frame_fig', 'clickData'),
    Input('reset-btn', 'n_clicks')]
)
def show_frame(hoverData, clickData, n_clicks):
    global prev_click, match_id, rally_id, frame_id, x_gt, y_gt, vis_gt, x_pred, y_pred, vis_pred, x_correct, y_correct, vis_correct
    trigger = dash.callback_context.triggered[0]["prop_id"]
    marker_size = 10
    traj_len = 9
    frame_id = hoverData['points'][0]['x']

    # Read frame image
    img_path = os.path.join(data_dir, split, f'match{match_id}', 'frame', f'{rally_id}', f'{frame_id}.{IMG_FORMAT}')
    assert os.path.exists(img_path), f'Image not found: {img_path}'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_fig = px.imshow(img)

    if prev_click != n_clicks:
        # Reset corrected label
        prev_click = n_clicks
        x_correct[frame_id] = x_gt[frame_id]
        y_correct[frame_id] = y_gt[frame_id]
        vis_correct[frame_id] = vis_gt[frame_id]

        frame_fig = go.Figure()
        frame_fig.add_trace(img_fig.data[0])
        frame_fig.add_trace(
            go.Scatter(x=x_gt[frame_id-int(traj_len/2):frame_id+int(traj_len/2)+1],
                       y=y_gt[frame_id-int(traj_len/2):frame_id+int(traj_len/2)+1],
                       marker_color=[f'rgba(255, {170+10*i}, 0, {0.3+0.05*i})' for i in range(9)],
                       text=[f for f in range(frame_id-int(traj_len/2), frame_id+int(traj_len/2)+1)],
                       mode='markers', marker_size=marker_size, name='neighbor')
        )
        frame_fig.add_trace(
            go.Scatter(x=x_pred[frame_id:frame_id+1], y=y_pred[frame_id:frame_id+1],
                       marker_color=['rgba(0, 255, 0, 0.5)'], text=[frame_id],
                       mode='markers', marker_size=marker_size, name='pred')
        )
        frame_fig.add_trace(
            go.Scatter(x=x_gt[frame_id:frame_id+1], y=y_gt[frame_id:frame_id+1],
                       marker_color=['rgba(255, 0, 0, 0.5)'], text=[frame_id],
                       mode='markers', marker_size=marker_size, name='gt')
        )
        frame_fig.update_layout(dragmode='pan', clickmode='event+select',
                                margin={'l':0, 'r':0, 't':0, 'b':0}, autosize=False, width=1280, height=720,
                                title_text=f'f_{frame_id} label: ({x_correct[frame_id]}, {y_correct[frame_id]})')
        frame_fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=0.5))
        return frame_fig
    if trigger == "frame_fig.clickData":
        # Show clicked point
        #print(f'click_data: {clickData}')
        click_x = clickData['points'][0]['x']
        click_y = clickData['points'][0]['y']
        x_correct[frame_id] = click_x
        y_correct[frame_id] = click_y
        vis_correct[frame_id] = 1 if click_x == 0 and click_y == 0 else 0
        
        frame_fig = go.Figure()
        frame_fig.add_trace(img_fig.data[0])
        frame_fig.add_trace(
            go.Scatter(x=x_gt[frame_id-int(traj_len/2):frame_id+int(traj_len/2)+1],
                       y=y_gt[frame_id-int(traj_len/2):frame_id+int(traj_len/2)+1],
                       marker_color=[f'rgba(255, {170+10*i}, 0, {0.3+0.05*i})' for i in range(9)],
                       text=[f for f in range(frame_id-int(traj_len/2), frame_id+int(traj_len/2)+1)],
                       mode='markers', marker_size=marker_size, name='neighbor')
        )
        frame_fig.add_trace(
            go.Scatter(x=x_pred[frame_id:frame_id+1], y=y_pred[frame_id:frame_id+1],
                       marker_color=['rgba(0, 255, 0, 0.5)'], text=[frame_id],
                       mode='markers', marker_size=marker_size, name='pred')
        )
        frame_fig.add_trace(
            go.Scatter(x=x_gt[frame_id:frame_id+1], y=y_gt[frame_id:frame_id+1],
                       marker_color=['rgba(255, 0, 0, 0.5)'], text=[frame_id],
                       mode='markers', marker_size=marker_size, name='gt')
        )
        frame_fig.add_trace(
            go.Scatter(x=[click_x], y=[click_y],
                       marker_color=['rgba(0, 0, 255, 1.)'], text=[frame_id],
                       mode='markers', marker_size=marker_size, name='correct')
        )
        frame_fig.update_layout(dragmode='pan', clickmode='event+select',
                                margin={'l':0, 'r':0, 't':0, 'b':0}, autosize=False, width=1280, height=720,
                                title_text=f'f_{frame_id} label: ({x_correct[frame_id]}, {y_correct[frame_id]})')
        frame_fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=0.5))
        return frame_fig
    if trigger == "time_fig.hoverData":
        # Show hovered frame with neighbor labels
        #print(f'hover_data: {hoverData}')
        selectedData = None

        frame_fig = go.Figure()
        frame_fig.add_trace(img_fig.data[0])
        frame_fig.add_trace(
            go.Scatter(x=x_gt[frame_id-int(traj_len/2):frame_id+int(traj_len/2)+1],
                       y=y_gt[frame_id-int(traj_len/2):frame_id+int(traj_len/2)+1],
                       marker_color=[f'rgba(255, {170+10*i}, 0, {0.3+0.05*i})' for i in range(9)],
                       text=[f for f in range(frame_id-int(traj_len/2), frame_id+int(traj_len/2)+1)],
                       mode='markers', marker_size=marker_size, name='neighbor')
        )
        frame_fig.add_trace(
            go.Scatter(x=x_pred[frame_id:frame_id+1], y=y_pred[frame_id:frame_id+1],
                       marker_color=['rgba(0, 255, 0, 0.5)'], text=[frame_id],
                       mode='markers', marker_size=marker_size, name='pred')
        )
        frame_fig.add_trace(
            go.Scatter(x=x_gt[frame_id:frame_id+1], y=y_gt[frame_id:frame_id+1],
                       marker_color=['rgba(255, 0, 0, 0.5)'], text=[frame_id],
                       mode='markers', marker_size=marker_size, name='gt')
        )
        frame_fig.update_layout(dragmode='pan', clickmode='event+select',
                                margin={'l':0, 'r':0, 't':0, 'b':0}, autosize=False, width=1280, height=720,
                                title_text=f'f_{frame_id} label: ({x_correct[frame_id]}, {y_correct[frame_id]})')
        frame_fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=0.5))
        return frame_fig
    else:
        raise PreventUpdate
    
    
if __name__ == '__main__':
   app.run_server(host=host, debug=debug)
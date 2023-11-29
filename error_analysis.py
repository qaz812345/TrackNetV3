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
parser.add_argument('--mode', type=str, default='point')
parser.add_argument('--host', type=str, default='127.0.0.1')
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()

split = args.split
mode = args.mode
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
        # {'label': label_name, 'value': json_path},
    ]
else:
    raise ValueError(f'Invalid split: {split}')


# Init global variables
pred_types = ['TP', 'TN', 'FP1', 'FP2', 'FN']
pred_types_map = {pred_type: i for i, pred_type in enumerate(pred_types)}
match_id, rally_id, frame_id = None, None, None
x_gt, y_gt = None, None
x_pred_1, y_pred_1 = None, None
x_pred_2, y_pred_2 = None, None

# Generatedrop down list values of rally id
rally_keys = []
rally_dirs = get_rally_dirs(data_dir, split)
rally_dirs = [os.path.join(data_dir, d) for d in rally_dirs]
for rally_dir in rally_dirs:
    file_format_str = os.path.join('{}', 'match{}', 'frame', '{}')
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
            html.Label(['Result 1:'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(eval_file_list, eval_file_list[0]['value'], id='eval-file-1-dropdown')
        ], style=dict(width='20%', margin='10px')),
        html.Div(children=[
            html.Label(['Result 2:'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(eval_file_list, eval_file_list[0]['value'], id='eval-file-2-dropdown')
        ], style=dict(width='20%', margin='10px')),
        html.Div(children=[
            html.Label(['Rally ID:'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(rally_keys, rally_keys[0], id='rally-key-dropdown')
        ], style=dict(width='20%', margin='10px'))
    ], style={'display':'flex', 'justify-content':'center', 'text-align':'center'}),
    # Time series plot
    html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='time_fig',
                figure=go.Figure().set_subplots(rows=2, cols=1),
                config={'scrollZoom':True}
            ),
        ], style=dict(width='90%')),
    ], style={'display':'flex', 'justify-content':'center', 'text-align':'center'}),
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
    [Input('eval-file-1-dropdown', 'value'),
    Input('eval-file-2-dropdown', 'value'),
    Input('rally-key-dropdown', 'value')]
)
def change_dropdown(eval_file_1, eval_file_2, rally_key):
    global match_id, rally_id, x_gt, y_gt, x_pred_1, y_pred_1, x_pred_2, y_pred_2
    
    # Bar chart settings
    bar_width = 1
    y_min, y_max = - 0.2, 1.5
    colors = {'TP': '#65AD6C', 'TN': '#D47D7D', 'FP1': 'green', 'FP2': 'red', 'FN': 'blue'}

    # Parse rally key
    rally_key_splits = rally_key.split('_')
    match_id, rally_id = rally_key_splits[0], '_'.join(rally_key_splits[1:])
    
    # Read prediction results
    print(f'File 1: {eval_file_1}')
    print(f'File 2: {eval_file_2}')
    eval_dict_1 = json.load(open(eval_file_1))['pred_dict'][rally_key]
    eval_dict_2 = json.load(open(eval_file_2))['pred_dict'][rally_key]
    x_pred_1, y_pred_1, vis_pred_1 = np.array(eval_dict_1['X']), np.array(eval_dict_1['Y']), np.array(eval_dict_1['Visibility'])
    x_pred_2, y_pred_2, vis_pred_2 = np.array(eval_dict_2['X']), np.array(eval_dict_2['Y']), np.array(eval_dict_2['Visibility'])

    # Parse prediction result into stack bar chart data
    bar_list = [dict() for _ in range(2)]
    for i, eval_dict in [(0, eval_dict_1), (1, eval_dict_2)]:
        for pred_type in pred_types:
            bar_list[i][pred_type] = (np.array(eval_dict['Type']) == pred_types_map[pred_type]).astype('int')
        bar_list[i]['Error'] = bar_list[i]['FN'] + bar_list[i]['FP1'] + bar_list[i]['FP2']
        bar_list[i]['TP'] = bar_list[i]['TP'] * y_min
        bar_list[i]['TN'] = bar_list[i]['TN'] * y_min
    
    # Read ground truth labels
    csv_dir = 'corrected_csv' if split == 'test' else 'csv'
    assert os.path.exists(os.path.join(data_dir, split, f'match{match_id}', csv_dir, f'{rally_id}_ball.csv'))
    csv_file = os.path.join(data_dir, split, f'match{match_id}', csv_dir, f'{rally_id}_ball.csv')
    label_df = pd.read_csv(csv_file, encoding='utf8')
    x_gt, y_gt, vis_gt = np.array(label_df['X']), np.array(label_df['Y']), np.array(label_df['Visibility'])
    
    # Time series plot
    timestamp = np.arange(len(label_df))
    hover_data = np.stack([x_gt, y_gt, vis_gt, x_pred_1, y_pred_1, vis_pred_1, x_pred_2, y_pred_2, vis_pred_2], axis=1)
    time_fig = go.Figure().set_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, subplot_titles=(eval_file_1, eval_file_2))
    for i, showlegend in [(0, True), (1, False)]:
        for pred_type in pred_types: 
            time_fig.add_trace(
                go.Bar(x=timestamp, y=bar_list[i][pred_type], customdata=hover_data,
                    width=bar_width, marker_color=colors[pred_type], name=pred_type, legendgroup=pred_type, showlegend=showlegend),
                row=i+1, col=1
            )
        # Visualize effective trajectory
        if split == 'test':
            # The moment of serve
            time_fig.add_vline(x=start_f[rally_key]-bar_width/2, line_width=1, line_dash='dash', line_color='gray', row=i+1, col=1)
            # The moment of the ball touch ground
            time_fig.add_vline(x=end_f[rally_key]-bar_width/2, line_width=1, line_dash='dash', line_color='gray', row=i+1, col=1)

        time_fig.update_yaxes(title_text='Error Count', range=[y_min, y_max], fixedrange=True, row=i+1, col=1)
    
    time_fig.update_xaxes(title_text='Frame ID', row=2, col=1)
    time_fig.update_layout(barmode='stack', dragmode='pan', clickmode='event+select', legend_title='Result Type',
                           margin={'l':20, 'r':20, 't':50, 'b':10}, height=300, title_text=f'Rally {rally_key} Error Distribution', title_x=0.5)
    time_fig.update_traces(
        hovertemplate="<br>".join([
            "frame id: %{x}",
            "label: ( %{customdata[0]}, %{customdata[1]} ), vis: %{customdata[2]}",
            "pred 1: ( %{customdata[3]}, %{customdata[4]} ), vis: %{customdata[5]}",
            "pred 2: ( %{customdata[6]}, %{customdata[7]} ), vis: %{customdata[8]}",
        ])
    )
    return time_fig


@app.callback(
    Output('frame_fig', 'figure'),
    [Input('time_fig', 'hoverData'),]
)
def show_frame(hoverData):
    global match_id, rally_id, x_gt, y_gt, x_pred_1, y_pred_1, x_pred_2, y_pred_2
    traj_len = 16
    radius, bbox_width, marker_size = 5, 1, 5
    point_visible = True if mode == 'point' else 'legendonly'
    traj_visible = True if mode == 'traj' else 'legendonly'
    
    #print(f'hover_data: {hoverData}')
    frame_id = hoverData['points'][0]['x']
    cx, cy = x_gt[frame_id], y_gt[frame_id]
    cx_pred_1, cy_pred_1 = x_pred_1[frame_id], y_pred_1[frame_id]
    cx_pred_2, cy_pred_2 = x_pred_2[frame_id], y_pred_2[frame_id]

    # Read Read frame image
    img_path = os.path.join(data_dir, split, f'match{match_id}', 'frame', rally_id, f'{frame_id}.{IMG_FORMAT}')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    img_fig = px.imshow(img)

    # Frame plot
    frame_fig = go.Figure()
    frame_fig.add_trace(img_fig.data[0])

    # Add button to show/hide bbox
    '''gt_bbox = [dict(type="rect", x0=cx-radius, y0=cy-radius, x1=cx+radius, y1=cy+radius, line=dict(color="red", width=bbox_width))]
    pred_bbox_1 = [dict(type="rect", x0=cx_pred_1-radius, y0=cy_pred_1-radius, x1=cx_pred_1+radius, y1=cy_pred_1+radius, line=dict(color="green", width=bbox_width))]
    pred_bbox_2 = [dict(type="rect", x0=cx_pred_2-radius, y0=cy_pred_2-radius, x1=cx_pred_2+radius, y1=cy_pred_2+radius, line=dict(color="blue", width=bbox_width))]
    frame_fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="All", method="relayout", args=["shapes", gt_bbox + pred_bbox_1 + pred_bbox_2]),
                    dict(label="GT (Red)", method="relayout", args=["shapes", gt_bbox]),
                    dict(label="Pred 1 (Green)", method="relayout", args=["shapes", pred_bbox_1]),
                    dict(label="Pred 2 (Blue)", method="relayout", args=["shapes", pred_bbox_2]),
                    dict(label="None", method="relayout", args=["shapes", []])
                ]
            )
        ]
    )'''
    
    frame_fig.add_trace(
        go.Scatter(x=x_pred_1[frame_id-traj_len+1:frame_id+1],
                    y=y_pred_1[frame_id-traj_len+1:frame_id+1],
                    marker_color=[f'rgba({170+int(80/traj_len)*i}, {170+int(80/traj_len)*i}, 0, 1)' for i in range(traj_len)],
                    text=[f for f in range(frame_id-int(traj_len/2), frame_id+int(traj_len/2)+1)],
                    mode='markers', marker_size=marker_size, name='pred 1_traj', visible=traj_visible)
    )
    frame_fig.add_trace(
        go.Scatter(x=x_pred_2[frame_id-traj_len+1:frame_id+1],
                    y=y_pred_2[frame_id-traj_len+1:frame_id+1],
                    marker_color=[f'rgba(0, {170+int(80/traj_len)*i}, 0, 1)' for i in range(traj_len)],
                    text=[f for f in range(frame_id-int(traj_len/2), frame_id+int(traj_len/2)+1)],
                    mode='markers', marker_size=marker_size, name='pred 2_traj', visible=traj_visible)
    )
    frame_fig.add_trace(
        go.Scatter(x=x_gt[frame_id-traj_len+1:frame_id+1],
                    y=y_gt[frame_id-traj_len+1:frame_id+1],
                    marker_color=[f'rgba({170+int(80/traj_len)*i}, 0, 0, 1)' for i in range(traj_len)],
                    text=[f for f in range(frame_id-int(traj_len/2), frame_id+int(traj_len/2)+1)],
                    mode='markers', marker_size=marker_size, name='gt traj', visible=traj_visible)
    )
    frame_fig.add_trace(
        go.Scatter(x=x_pred_1[frame_id:frame_id+1], y=y_pred_1[frame_id:frame_id+1],
                    marker_color=['rgba(0, 0, 255, 1)'], text=[frame_id],
                    mode='markers', marker_size=marker_size, name='pred 1', visible=point_visible)
    )
    frame_fig.add_trace(
        go.Scatter(x=x_pred_2[frame_id:frame_id+1], y=y_pred_2[frame_id:frame_id+1],
                    marker_color=['rgba(0, 255, 0, 1)'], text=[frame_id],
                    mode='markers', marker_size=marker_size, name='pred 2', visible=point_visible)
    )
    frame_fig.add_trace(
        go.Scatter(x=x_gt[frame_id:frame_id+1], y=y_gt[frame_id:frame_id+1],
                    marker_color=['rgba(255, 0, 0, 1)'], text=[frame_id],
                    mode='markers', marker_size=marker_size, name='gt', visible=point_visible)
    )
    frame_fig.update_layout(dragmode='pan', clickmode='event+select', autosize=False,
                            margin={'l':0, 'r':0, 't':50, 'b':0}, width=img_w, height=img_h,
                            title_text=f'Frame {frame_id} label: ({cx}, {cy}), pred 1: ({cx_pred_1}, {cy_pred_1}), pred 2: ({cx_pred_2}, {cy_pred_2})', title_x=0.45)

    return frame_fig

if __name__ == '__main__':
   app.run_server(host=host, debug=debug)
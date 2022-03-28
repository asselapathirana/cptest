import base64
import os

from flask import Flask, send_from_directory
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import tempfile
#import FAKEpredict as predict
import predict
import numpy as np
import cv2

SIDE = 227

server = Flask(__name__)
app = dash.Dash(__name__,server=server)
# apprantly it is critical to provide name to Dash constructor too. 

# load the CNN 
graph, cnn_model, cnn_lb = predict.load_model_and_labels('./data/concrete_best.model','./data/concrete_lb.pickle')


app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        className="uploadArea",
        # Allow multiple files to be uploaded
        multiple=True,
        accept="image/*",
        max_size="2100000", # 2 MB
        
    ),
    html.Div(id='image_ul'),
])


def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def process_image(fn, content):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    data = data_uri_to_cv2_img(content) #endode if needed 
    #decode if needed dc=base64.decodebytes(data)
    t,ext=os.path.splitext(fn)    
    #output = predict.predict(graph, cnn_model, cnn_lb, data, 64, 64)	
    preds = predict.predict2(graph, cnn_model, data, SIDE, SIDE)
    i = preds.argmax(axis=1)[0]
    label = cnn_lb.classes_[i]
    # draw the class label + probability on the output image
    if i==0:
        bd = '5px solid green'
    else:
        bd = '5px solid red'

    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)   
    img_=html.Img(src=content, className="scaledimage",)
    img = html.Div(img_, className="imagediv")
    div = html.Div(text, className="textdiv")
    return html.Div(children=[div, img,],
                    style={"border" : bd, }, #, "border-radius": "5px", "width": "30vw",  "height": "30vw", "margin":"5px", "box-sizing": "border-box"},
                    className="three columns")
                             

@app.callback(
    Output("image_ul", "children"),
              [Input('upload-data', 'filename'),
               Input('upload-data', 'contents'),
              Input('upload-data', 'last_modified')])
def update_output(uploaded_filename, uploaded_file_contents, modified_date):
    """Save uploaded files and regenerate the file list."""

    if uploaded_filename is not None and uploaded_file_contents is not None:
        if isinstance(uploaded_filename, str): # its a list of files
            uploaded_filename = [uploaded_filename]
            uploaded_file_contents = [uploaded_file_contents]
            modified_date = [modified_date]
        res=[]
        for file, content in zip(uploaded_filename,uploaded_file_contents):
            r_ = process_image(file, content)
            res.append(r_)
        return html.Div(id="results", children=res, className="row")
                


if __name__ == "__main__":
    import os
    if 'WINGDB_ACTIVE' in os.environ:
        server.debug = False
        server.run(port=8888)

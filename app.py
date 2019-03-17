import base64
import os
from urllib.parse import quote as urlquote

from flask import Flask, send_from_directory
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import tempfile
import predict
import numpy as np
import cv2

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(server=server, external_stylesheets=external_stylesheets)

#UPLOAD_DIRECTORY = "/project/app_uploaded_files"

# load the CNN 

graph, cnn_model, cnn_lb = predict.load_model_and_labels('concrete.model','concrete_lb.pickle')
#if not os.path.exists(UPLOAD_DIRECTORY):
#    os.makedirs(UPLOAD_DIRECTORY)

#@server.route("/download/<path:path>")
#def download(path):
#    """Serve a file from the upload directory."""
#    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True,
        accept="image/*",
        max_size="2100000", # 2 MB
        
    ),
    html.Div(id='image_ul'),
])



#def uploaded_files():
    #"""List the files in the upload directory."""
    #files = []
    #for filename in os.listdir(UPLOAD_DIRECTORY):
        #path = os.path.join(UPLOAD_DIRECTORY, filename)
        #if os.path.isfile(path):
            #files.append(filename)
    #return files


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
    preds = predict.predict2(graph, cnn_model, data, 64, 64)
    i = preds.argmax(axis=1)[0]
    label = cnn_lb.classes_[i]
    # draw the class label + probability on the output image
    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)    
    return html.Div(text), html.Img(src=content)


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
            #fix this to process all files are return
            # try https://dash.plot.ly/datatable 
            res1, res2 = process_image(file, content)
            res.append(html.Table([html.Tr([html.Td(res1)]),
                               html.Tr([html.Td(res2)])])
                       )
        return html.Div(id="results", children=res)
                


if __name__ == "__main__":
    import os
    if 'WINGDB_ACTIVE' in os.environ:
        server.debug = False
        server.run(port=8888)

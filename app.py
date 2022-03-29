mport base64
import os

from flask import Flask
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import tempfile
#import FAKEpredict as predict
import predict
import numpy as np
import cv2
import os

BASE_PATH = os.getenv("DASH_BASE_PATHNAME","/")
SIDE = 227

server = Flask(__name__)
# apprantly it is critical to provide name to Dash constructor too. 
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN], 
                server=server, url_base_pathname=BASE_PATH)



# load the CNN 
print ("Loadeding")
graph, cnn_model, cnn_lb = predict.load_model_and_labels('./data/concrete_best.model','./data/concrete_lb.pickle')
print ("Loaded")

_INP = dcc.Upload(
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
        
    )

_OUTP = html.Div(id='image_ul')



app.layout = dbc.Container([
    dbc.Row(#dbc.Card(
        [html.H1("Concrete Crack Detection Using Artificial Intlligence",),
         dbc.Alert(
                     "Drag and drop some images (for best results use images of 200x200 pixels or smaller) and see how the AI classifies them. Positive: Likely cracks. Negative: Likely no cracks.",
                     id="alert-fade",
                     dismissable=True,
                     is_open=True,
                     color='info'
        )         
         ],
         align="center"
        ),
    dbc.Row([
        dbc.Col(dbc.Container(dbc.Card([_INP]))),
        ]),
    dbc.Row([
        dbc.Col(dbc.Container(dbc.Card([_OUTP]))),
        ]),    
        
])




def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
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
    else:
        server.run()
>>>>>>> 9f3f8fa08344845dc549caa58a669d62e14eb776

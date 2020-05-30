from flask import Flask,render_template,request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
#Loading the model to compute the result
model=pickle.load(open('model.pkl','rb'))
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/result',methods=['POST'])
def result():
    rm=request.form['Radius_mean']  
    tm=request.form['Texture_mean']
    sm=request.form['Smoothness_mean']  
    cm=request.form['Compactness_mean']
    sym=request.form['Symmetry_mean']  
    fdm=request.form['Fractal_Dimension_mean']
    rse=request.form['Radius_se']  
    sse=request.form['Smoothness_se']  
    cse=request.form['Compactness_se']
    fdse=request.form['Fractal_Dimension_se'] 
    rw=request.form['Radius_worst']  
    tw=request.form['Texture_worst']
    sw=request.form['Smoothness_worst']  
    cw=request.form['Compactness_worst']
    syw=request.form['Symmetry_worst']  
    fdw=request.form['Fractal_Dimension_worst'] 
    data={'radius_mean':rm,'texture_mean':tm,'smoothness_mean':sm,'compactness_mean':cm,'symmetry_meam':sym,'fractal_dimension_mean':fdm,'radius_se':rse,'smoothness_se':sse,'compactness_se':cse,'fractal_dimension_se':fdse,'radius_worst':rw,'texture_worst':tw,'smoothness_worst':sw,'compactness_worst':cw,'symmetry_worst':syw,'fractal_dimension_worst':fdw}
    x=pd.DataFrame(data,index=['1'])
    x=x.astype(float)
    sc=StandardScaler()
    x=sc.fit_transform(x)
    y_predict=model.predict(x)
    y=y_predict.item()
    if y==0:
        output='Benign'
    else:
        output='Malignant'
    return render_template('predict.html',prediction_text1="As per the results from your FNA test,your cell is {}.".format(output),prediction_text2="Please do consult a doctor for further treatments.")
@app.route('/predict')
def predict():
    return render_template('predict.html')
@app.route('/about')
def about():
    return render_template('about.html')


if __name__=="__main__":
    app.run()
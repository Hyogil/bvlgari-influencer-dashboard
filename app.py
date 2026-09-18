from pathlib import Path
import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate

BASE=Path(__file__).resolve().parent; DATA=BASE/'data'; DATA.mkdir(exist_ok=True)
TARGET='simulated_campaign_success_target'
FEATURES=['past_luxury_campaign_success_rate_sim','avg_engagement_rate','luxury_post_ratio_sim','past_campaign_count_sim','follower_reach_log','past_campaign_success_rate_sim']
FEATURE_LABELS={
 'past_luxury_campaign_success_rate_sim':'Past Luxury Campaign Success Rate',
 'avg_engagement_rate':'Average Engagement Rate',
 'luxury_post_ratio_sim':'Luxury Post Ratio',
 'past_campaign_count_sim':'Past Campaign Count',
 'follower_reach_log':'Follower Reach (log)',
 'past_campaign_success_rate_sim':'Past Campaign Success Rate'}
app=FastAPI(title='Influencer Success Prediction Dashboard')
app.mount('/static',StaticFiles(directory=BASE/'static'),name='static'); templates=Jinja2Templates(directory=BASE/'templates')
STATE={'models':{},'train_rows':0,'test':None,'predictions':{},'metrics':{},'train_metrics':{},'formulas':{}}

def read_xlsx(path):
    xls=pd.ExcelFile(path); sheet='Creators_Enriched' if 'Creators_Enriched' in xls.sheet_names else xls.sheet_names[0]
    return pd.read_excel(path,sheet_name=sheet)

def add_derived(df):
    df=df.copy()
    if 'followers' in df.columns:
        followers=pd.to_numeric(df['followers'],errors='coerce').clip(lower=0)
        df['follower_reach_log']=np.log1p(followers)
    return df

def clean_xy(df,require_target=True):
    df=add_derived(df); missing=[c for c in FEATURES if c not in df.columns]
    if missing: raise ValueError('Missing feature columns: '+', '.join(missing))
    X=df[FEATURES].apply(pd.to_numeric,errors='coerce')
    if not require_target:return X,None
    if TARGET not in df.columns:raise ValueError(f'Missing target column: {TARGET}')
    valid=df[TARGET].notna()
    if valid.sum()==0:raise ValueError(f'Validation target is empty. "{TARGET}" must contain actual 0/1 outcomes. The prediction/test file should be masked, but the validation file must NOT be masked.')
    y=pd.to_numeric(df.loc[valid,TARGET],errors='coerce')
    if y.isna().any():raise ValueError('Validation target contains non-numeric values. Use only 0 (Fail) and 1 (Success).')
    if not set(y.astype(int).unique()).issubset({0,1}):raise ValueError('Validation target must contain only 0 (Fail) and 1 (Success).')
    return X.loc[valid],y.astype(int)

def make_models():
    return {
      'logistic':Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42))]),
      'tree':Pipeline([('imputer',SimpleImputer(strategy='median')),('model',DecisionTreeClassifier(max_depth=5,min_samples_leaf=20,class_weight='balanced',random_state=42))])}

def safe_float(v):
    if v is None or (isinstance(v,float) and math.isnan(v)):return None
    return float(v)

def evaluate(model,df):
    X,y=clean_xy(df,True); p=model.predict_proba(X)[:,1]; pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'accuracy':safe_float(accuracy_score(y,pred)),'precision':safe_float(precision_score(y,pred,zero_division=0)),'recall':safe_float(recall_score(y,pred,zero_division=0)),'f1':safe_float(f1_score(y,pred,zero_division=0)),'auc':safe_float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,'brier':safe_float(brier_score_loss(y,p)),'confusion':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)},'n':int(len(y))}

def cv_metrics(models,X,y):
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42); out={}
    for name,m in models.items():
        r=cross_validate(m,X,y,cv=cv,scoring={'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1','auc':'roc_auc','brier':'neg_brier_score'},error_score='raise')
        out[name]={k:float(np.mean(r['test_'+k])) for k in ['accuracy','precision','recall','f1','auc']}; out[name]['brier']=float(-np.mean(r['test_brier']))
    return out

def training_formula(models):
    lp=models['logistic']; scaler=lp.named_steps['scale']; clf=lp.named_steps['model']; details=[]; terms=[]
    for name,b,mean,scale in zip(FEATURES,clf.coef_[0],scaler.mean_,scaler.scale_):
        terms.append(f'({b:+.6f} × z_{name})'); details.append({'feature':name,'label':FEATURE_LABELS[name],'coefficient':float(b),'mean':float(mean),'scale':float(scale)})
    logistic={'title':'Logistic Regression','equation':f'z = {float(clf.intercept_[0]):.6f} '+' '.join(terms),'probability':'f(x) = P(Success=1 | x) = 1 / (1 + e^(-z))','threshold':'Predicted Success if f(x) ≥ 0.500; otherwise Fail.','details':details}
    tree_model=models['tree'].named_steps['model']; rules=export_text(tree_model,feature_names=[FEATURE_LABELS[x] for x in FEATURES],decimals=4)
    tree={'title':'Decision Tree','equation':'f(x) = P(Success=1 | terminal leaf reached by x)','probability':'The six features follow learned IF/ELSE splits; the terminal leaf class proportion is the success probability.','threshold':'Predicted Success if f(x) ≥ 0.500; otherwise Fail.','rules':rules}
    return {'logistic':logistic,'tree':tree}

def model_assessment(metrics):
    L,T=metrics['logistic'],metrics['tree']
    # Primary: ROC-AUC (discrimination); secondary: Brier (probability calibration); then F1 and accuracy.
    aucL,aucT=L.get('auc'),T.get('auc'); reasons=[]
    if aucL is not None and aucT is not None and abs(aucL-aucT)>=0.01:
        winner='logistic' if aucL>aucT else 'tree'; reasons.append(f'Primary criterion ROC-AUC: {aucL:.3f} vs {aucT:.3f}.')
    elif abs(L['brier']-T['brier'])>=0.005:
        winner='logistic' if L['brier']<T['brier'] else 'tree'; reasons.append(f'ROC-AUC is close; lower Brier score is used as the probability-calibration tie-breaker ({L["brier"]:.3f} vs {T["brier"]:.3f}).')
    elif abs(L['f1']-T['f1'])>=0.01:
        winner='logistic' if L['f1']>T['f1'] else 'tree'; reasons.append(f'ROC-AUC and Brier are close; F1 is used next ({L["f1"]:.3f} vs {T["f1"]:.3f}).')
    else:
        winner='logistic' if L['accuracy']>=T['accuracy'] else 'tree'; reasons.append('The models are very close on discrimination/calibration; accuracy is used only as a final tie-breaker.')
    w='Logistic Regression' if winner=='logistic' else 'Decision Tree'
    return {'winner':winner,'winner_label':w,'summary':f'{w} is preferred on this validation set under the predefined evaluation hierarchy.','reason':' '.join(reasons),'method':'Independent validation; primary ROC-AUC, secondary Brier score, then F1 and Accuracy. Lower Brier is better; higher values are better for the other metrics.','caution':'This conclusion applies to the supplied validation sample and should not be interpreted as universal model superiority.'}

def predict_rows(model,df):
    X,_=clean_xy(df,False); probs=model.predict_proba(X)[:,1]; out=[]
    for pos,(_,r) in enumerate(df.iterrows()):
        p=float(probs[pos]); out.append({'rank':0,'handle':str(r.get('handle','')),'name':str(r.get('name','')),'platform':str(r.get('platform','')),'niche':str(r.get('niche','')),'followers':safe_float(r.get('followers')),'engagement':safe_float(r.get('avg_engagement_rate')),'probability':p,'label':'Success' if p>=.5 else 'Fail','image_id':str(r.get('handle','')) if pd.notna(r.get('handle')) else ''})
    out.sort(key=lambda z:z['probability'],reverse=True)
    for i,r in enumerate(out,1):r['rank']=i
    return out

@app.get('/',response_class=HTMLResponse)
def home(request:Request):return templates.TemplateResponse('index.html',{'request':request})
@app.post('/api/train')
async def train(file:UploadFile=File(...)):
    try:
        path=DATA/'train_uploaded.xlsx'; path.write_bytes(await file.read()); df=read_xlsx(path); X,y=clean_xy(df,True); models=make_models(); cvm=cv_metrics(models,X,y)
        for m in models.values():m.fit(X,y)
        STATE.update({'models':models,'train_rows':len(X),'train_metrics':cvm,'formulas':training_formula(models),'metrics':{},'predictions':{}})
        return {'ok':True,'rows':len(X),'features':[FEATURE_LABELS[x] for x in FEATURES],'positive_rate':float(y.mean()),'train_metrics':cvm,'formulas':STATE['formulas']}
    except Exception as e:raise HTTPException(400,str(e))
@app.post('/api/load-test')
async def load_test(file:UploadFile=File(...)):
    try:
        path=DATA/'test_uploaded.xlsx'; path.write_bytes(await file.read()); df=read_xlsx(path); clean_xy(df,False); STATE['test']=df; STATE['predictions']={}; masked=TARGET not in df.columns or df[TARGET].isna().all(); return {'ok':True,'rows':len(df),'target_masked':bool(masked)}
    except Exception as e:raise HTTPException(400,str(e))
@app.post('/api/predict')
def predict():
    if not STATE['models']:raise HTTPException(400,'Train the models first.')
    if STATE['test'] is None:raise HTTPException(400,'Load the test dataset first.')
    STATE['predictions']={k:predict_rows(m,STATE['test']) for k,m in STATE['models'].items()}; return {'ok':True,'rows':len(STATE['test']),'predictions':STATE['predictions']}
@app.post('/api/validate')
async def validate(file:UploadFile=File(...)):
    if not STATE['models']:raise HTTPException(400,'Train the models first.')
    try:
        path=DATA/'validation_uploaded.xlsx'; path.write_bytes(await file.read()); df=read_xlsx(path); Xv,yv=clean_xy(df,True)
        if STATE['test'] is not None and len(df)!=len(STATE['test']):raise ValueError(f'Row-count mismatch: test has {len(STATE["test"])} rows but validation has {len(df)} rows.')
        metrics={k:evaluate(m,df) for k,m in STATE['models'].items()}; STATE['metrics']=metrics
        actual_by_handle={str(r.get('handle','')):int(a) for (_,r),a in zip(df.loc[df[TARGET].notna()].iterrows(),yv.tolist())}
        for model_rows in STATE['predictions'].values():
            for row in model_rows:
                a=actual_by_handle.get(row['handle']); row['actual']=('Success' if a==1 else 'Fail') if a is not None else None; row['match']=(row['label']==row['actual']) if row['actual'] else None
        return {'ok':True,'rows':len(yv),'metrics':metrics,'predictions':STATE['predictions'],'assessment':model_assessment(metrics)}
    except Exception as e:raise HTTPException(400,str(e))

from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .engine import destinations, encode, load_model, explain
import numpy as np

app=FastAPI(title='Trip Decider API',version='1.0.0',description='DNN powered Indian destination recommendations')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['*'],allow_headers=['*'])
class Preferences(BaseModel):
    starting_city:str=Field(min_length=2,max_length=80)
    budget:int=Field(ge=1000,le=500000)
    days:int=Field(ge=1,le=30)
    types:List[str]=[]
    weather:str='any'
    companions:str='solo'
    style:str='balanced'
    activities:List[str]=[]
    month:int=Field(ge=1,le=12)
    max_travel_hours:Optional[int]=None
    accessibility:bool=False
    avoid_crowds:bool=False
    hill_station:bool=False
    road_trip:bool=False

@app.get('/api/health')
def health():
    return {'status':'ok','destinations':len(destinations()),'model_available':Path(__file__).resolve().parents[1].joinpath('models/trip_decider_model.keras').exists()}
@app.get('/api/destinations')
def list_destinations():
    return destinations().fillna('').to_dict(orient='records')
@app.get('/api/destinations/{name}')
def destination_detail(name:str):
    df=destinations(); rows=df[df.name.str.lower()==name.lower()]
    if rows.empty: raise HTTPException(404,'Destination not found')
    return rows.iloc[0].fillna('').to_dict()
@app.post('/api/recommendations')
def recommendations(p:Preferences):
    try: model=load_model(); df=destinations()
    except FileNotFoundError as e: raise HTTPException(503,str(e))
    user=p.dict(); X=np.stack([encode(user,d) for _,d in df.iterrows()]); scores=model.predict(X,verbose=0).reshape(-1)
    results=[]
    for (_,d),score in zip(df.iterrows(),scores):
        estimated=int((float(d.min_cost)+float(d.max_cost))/2)
        results.append({'name':d['name'],'state':d['state'],'region':d['region'],'description':d['description'],'types':str(d['types']).split('|'),'climate':d['climate'],'activities':str(d['activities']).split('|'),'min_cost':int(d['min_cost']),'max_cost':int(d['max_cost']),'estimated_cost':estimated,'min_days':int(d['min_days']),'max_days':int(d['max_days']),'best_months':[int(x) for x in str(d['best_months']).split('|')],'score':round(float(score)*100,1),'explanations':explain(user,d),'latitude':float(d.latitude),'longitude':float(d.longitude),'budget_fit':int(d.min_cost)<=p.budget,'duration_fit':int(d.min_days)<=p.days<=int(d.max_days)})
    results.sort(key=lambda x:x['score'],reverse=True)
    eligible=[item for item in results if item['budget_fit'] and item['duration_fit']]
    # Keep strict matches when there are enough choices; otherwise include near matches
    # with explicit budget/duration notes in their card explanations.
    ranked=eligible if len(eligible)>=5 else results
    return {'recommendations':ranked[:10],'eligibility_fallback':len(eligible)<5,'model_note':'Suitability is a model-generated score based on synthetic training labels; it is not a verified probability of enjoyment.'}
@app.post('/api/budget-estimate')
def budget_estimate(payload:dict):
    try: days=max(1,int(payload.get('days',1))); daily=int(payload.get('daily_budget',1500)); travel=int(payload.get('travel_cost',1000)); stay=int(payload.get('stay_cost',1000)); activities=int(payload.get('activities_cost',500))
    except (TypeError,ValueError): raise HTTPException(422,'Budget values must be whole numbers')
    return {'total':travel+days*(daily+stay+activities),'per_day':daily+stay+activities,'days':days}
@app.post('/api/compare')
def compare(payload:dict):
    names=payload.get('names',[])
    if not 2<=len(names)<=3: raise HTTPException(422,'Choose two or three destinations to compare')
    df=destinations(); items=[]
    for name in names:
        rows=df[df.name.str.lower()==str(name).lower()]
        if rows.empty: raise HTTPException(404,f'Destination not found: {name}')
        items.append(rows.iloc[0].fillna('').to_dict())
    return {'destinations':items}

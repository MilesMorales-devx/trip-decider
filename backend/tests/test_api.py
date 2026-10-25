import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app.main import app
from app.engine import destinations, encode
client=TestClient(app)
def test_health_and_dataset():
    assert client.get('/api/health').status_code==200
    assert len(client.get('/api/destinations').json())>=50
def test_detail_and_missing():
    assert client.get('/api/destinations/Manali').status_code==200
    assert client.get('/api/destinations/nope').status_code==404
def test_validation():
    assert client.post('/api/recommendations',json={}).status_code==422
def test_budget_and_compare():
    assert client.post('/api/budget-estimate',json={'days':3,'daily_budget':1000,'stay_cost':500,'travel_cost':1000,'activities_cost':500}).json()['total']==7000
    assert client.post('/api/compare',json={'names':['Goa','Manali']}).status_code==200
def test_preprocessing_repeatable_and_bounded():
    d=destinations().iloc[0]; u={'budget':10000,'days':4,'month':6,'types':['mountains'],'activities':['trekking'],'weather':'cool','style':'budget','companions':'solo'}
    assert (encode(u,d)==encode(u,d)).all()
    from app.engine import score_label
    assert 0<=score_label(u,d)<=1
def test_recommendation_uses_model():
    from app.engine import MODEL
    if not MODEL.exists():
        assert client.post('/api/recommendations',json={'starting_city':'Delhi','budget':10000,'days':4,'month':6}).status_code==503
    else:
        response=client.post('/api/recommendations',json={'starting_city':'Delhi','budget':10000,'days':4,'month':6})
        assert response.status_code==200 and len(response.json()['recommendations'])==10

def test_empty_or_invalid_dataset_is_rejected(tmp_path, monkeypatch):
    import app.engine as engine
    bad=tmp_path/'bad.csv'; bad.write_text('name,state\n')
    monkeypatch.setattr(engine,'DATA',bad)
    try:
        engine.destinations()
        assert False, 'invalid dataset should be rejected'
    except RuntimeError:
        pass

def test_missing_artifact_is_explicit(tmp_path, monkeypatch):
    import app.engine as engine
    monkeypatch.setattr(engine,'MODEL',tmp_path/'absent.keras')
    try:
        engine.load_model()
        assert False, 'missing artifact should raise'
    except FileNotFoundError as e:
        assert 'train_model.py' in str(e)

def test_model_inference_is_repeatable_and_bounded():
    from app.engine import MODEL, load_model
    if not MODEL.exists():
        import pytest; pytest.skip('Train the model to run inference checks')
    import numpy as np
    d=destinations().iloc[0]; x=encode({'budget':10000,'days':4,'month':6,'types':['mountains'],'activities':['trekking'],'weather':'cool','style':'budget','companions':'solo'},d)[None,:]
    model=load_model(); a=float(model.predict(x,verbose=0)[0][0]); b=float(model.predict(x,verbose=0)[0][0])
    assert 0<=a<=1 and a==b

def test_eligibility_fallback_discloses_mismatches():
    response=client.post('/api/recommendations',json={'starting_city':'Delhi','budget':1000,'days':1,'month':6})
    assert response.status_code==200
    payload=response.json()
    assert payload['eligibility_fallback'] is True
    assert payload['recommendations']
    assert any('may exceed' in ' '.join(r['explanations']) for r in payload['recommendations'])

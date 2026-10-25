"""Generate reproducible preference/destination labels and train the DNN."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app.engine import destinations, encode, score_label, MODEL, SCHEMA


def main():
    import tensorflow as tf
    np.random.seed(42); tf.random.set_seed(42)
    df=destinations(); rng=np.random.default_rng(42); X=[]; y=[]
    type_options=['mountains','beaches','nature','historical','adventure','spiritual','city','relaxation','wildlife','culture']
    activity_options=['trekking','sightseeing','photography','food','water activities','camping','wildlife','shopping','historical','relaxation']
    for _ in range(5000):
        u={'budget':int(rng.choice([4000,7500,15000,30000,50000])), 'days':int(rng.choice([2,4,7,11])),
           'month':int(rng.integers(1,13)), 'types':list(rng.choice(type_options,size=int(rng.integers(1,4)),replace=False)),
           'activities':list(rng.choice(activity_options,size=int(rng.integers(1,5)),replace=False)),
           'weather':str(rng.choice(['cool','pleasant','warm','tropical','any'])),
           'style':str(rng.choice(['budget','balanced','comfort','luxury'])), 'companions':str(rng.choice(['solo','couple','friends','family','group']))}
        for _,d in df.iterrows(): X.append(encode(u,d)); y.append(score_label(u,d))
    X=np.asarray(X); y=np.asarray(y,dtype=np.float32)
    xtr,xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    model=tf.keras.Sequential([tf.keras.layers.Input(shape=(X.shape[1],)),tf.keras.layers.Dense(128,activation='relu'),tf.keras.layers.BatchNormalization(),tf.keras.layers.Dropout(.25),tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dropout(.2),tf.keras.layers.Dense(32,activation='relu'),tf.keras.layers.Dense(1,activation='sigmoid')])
    model.compile(optimizer='adam',loss='mse',metrics=['mae'])
    model.fit(xtr,ytr,validation_split=.1,epochs=12,batch_size=256,verbose=2,callbacks=[tf.keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)])
    pred=model.predict(xte,verbose=0).ravel(); MODEL.parent.mkdir(parents=True,exist_ok=True); model.save(MODEL)
    SCHEMA.write_text(json.dumps({'feature_count':int(X.shape[1]),'types':['mountains','beaches','nature','historical','adventure','spiritual','city','relaxation','wildlife','culture','food'],'activities':['trekking','sightseeing','photography','food','water activities','camping','wildlife','shopping','historical','relaxation'],'seed':42},indent=2))
    metrics={'train_samples':len(xtr),'test_samples':len(xte),'mae':float(mean_absolute_error(yte,pred)),'rmse':float(mean_squared_error(yte,pred)**.5),'r2':float(r2_score(yte,pred))}
    (MODEL.parent/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()

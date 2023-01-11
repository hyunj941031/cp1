from flask import Flask, render_template, request, redirect, url_for
import sys
application = Flask(__name__)


@application.route("/")
def index():
    return render_template('index.html')

@application.route("/upload")
def upload():
    return render_template('upload.html')

@application.route("/uploaded", methods=['POST'])
def uploaded():
    f = request.files['file']
    f.save('static/data/{}.csv'.format(1))    
    return redirect(url_for('result'))

@application.route("/result")
def result():
    # 필요 라이브러리
    import pandas as pd
    import numpy as np
    import os
    
    # 파라미터 설정(minibatch 크기)
    minibatch = 4
    
    # 데이터 불러오기
    df = pd.read_csv('static/data/1.csv')
    
    # 독립변수 정규화
    df_scaled = df.copy()
    for c in range(len(df_scaled.iloc[0])-1):
        l = df_scaled.iloc[:, c].max()
        s = df_scaled.iloc[:, c].min()
        for r in range(len(df_scaled)):
            df_scaled.iloc[r, c] = (df_scaled.iloc[r, c] - s) / (l - s)
            
    # 무작위로 train / test 데이터셋 나누기(train=80%, test=20%)
    train = df_scaled.sample(frac=0.8, replace=False)
    train_index = train.index
    test = df_scaled.drop(index=train_index)
    
    # 독립변수(X)와 종속변수(y) column 나누기
    target = 'y'
    features = df_scaled.drop(columns=[target]).columns
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    # 파라미터, 편향
    # train 데이터에서 종속변수 y와의 상관관계에 따라 가중치 부여
    W11, W12, W13, W14, W15, W16, W17, W18 = train.corr()['y'][:-1]
    W = np.array([W11, W12, W13, W14, W15, W16, W17, W18])
    # 편향
    b = - (W.max() - (W.max() - W.min()) / 4)
    # minibatch마다 평가 결과 저장
    scores = []
    losses = []
    
    # minibatch로 나누기(4분할)
    for i in range(len(train)//minibatch):
        # minibatch만큼의 데이터의 인덱스(시작, 끝)
        mini_start = int(len(train)*i/(len(train)//minibatch))
        mini_end = int(len(train)*(i+1)/(len(train)//minibatch))
        # 독립변수와 가중치행렬을 내적
        k = X_train[mini_start:mini_end].dot(W)
        res = k + b
        # sigmoid함수 적용
        res = 1 / (1 + np.exp(-res))
        # res값이 0.5 이상은 1, 미만은 0으로 예측
        pred = np.where(res>=0.5, 1, 0)
        # 실제값
        real = y_train[mini_start:mini_end].reset_index(drop=True)
        # 예측값과 실제값 비교
        score = 0
        for m in range(mini_end-mini_start):
            if pred[m] == real[m]:
                score += 1
        # minibatch의 정확도(평균)
        scores.append(score/(m+1))
        
        # 손실함수 계산(교차 엔트로피)
        delta = 1e-7
        losses.append(-np.sum(real * np.log(pred+delta)))

    # 전체 정확도(평균)
    train_accuracy = np.mean(scores)
    # 전체 손실함수(평균)
    train_loss = np.mean(losses).round(3)
    
    # test셋 정확도 및 손실함수
    # 모델에 test데이터 입력
    k = X_test.dot(W)
    res = k + b
    res = 1 / (1 + np.exp(-res))
    pred = np.where(res>=0.5, 1, 0)
    # 실제값
    real = y_test.reset_index(drop=True)
    # test 예측값, 실제값 비교
    score = 0
    for m in range(len(X_test)):
        if pred[m] == real[m]:
            score += 1
    
    test_accuracy = score / len(X_test)
    test_loss = -np.sum(real * np.log(pred + delta)).round(3)
    
    os.remove('static/data/1.csv')
    return f"""
    [Epoch 1] TrainData - Loss = {train_loss}, Accuracy = {train_accuracy}
              TestData  - Loss = {test_loss},  Accuracy = {test_accuracy}
    """

if __name__ == "__main__":
    application.run(host='0.0.0.0')

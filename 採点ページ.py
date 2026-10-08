"""学生がCSVをアップロードすると即時に採点する。答案はサーバーだけに置く。"""
from pathlib import Path
import csv
import os
import io
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
    precision_score, f1_score, confusion_matrix, roc_curve)

PASS_AUC = 0.74
ANSWER_FILE = Path(os.environ.get('NHANES_ANSWER_FILE', str(Path(__file__).with_name('public_test_answer.csv'))))

st.set_page_config(page_title='医療AI演習・提出CSVの採点', layout='centered')
st.title('医療AI演習・提出CSVの採点')
st.write('submission.csvを選ぶと、自動で点数と合否が表示されます。結果をダウンロードしてLMSへ提出してください。')
st.caption(f'合格：ROC-AUC {PASS_AUC:.2f}以上。提出の時間・期間・回数に制限はありません。')

try:
    if ANSWER_FILE.exists():
        answer = pd.read_csv(ANSWER_FILE, dtype={'participant_id': str})
    else:
        answer = pd.read_csv(io.StringIO(st.secrets['answer_csv']), dtype={'participant_id': str})
    if list(answer.columns) != ['participant_id', 'target'] or answer.isna().any().any():
        raise ValueError()
    if not answer.participant_id.is_unique or set(answer.target.unique()) != {0, 1}:
        raise ValueError()
except Exception:
    st.error('採点の準備ができていません。教員に連絡してください。')
    st.stop()


def read_submission(content):
    if len(content) > 1_000_000:
        raise ValueError('CSVは1 MB以下にしてください。')
    try:
        text = content.decode('utf-8-sig')
        rows = list(csv.reader(io.StringIO(text), strict=True))
    except (UnicodeError, csv.Error):
        raise ValueError('UTF-8のCSVを選んでください。区切り文字や引用符も確認してください。') from None
    if not rows or rows[0] != ['participant_id', 'probability']:
        raise ValueError('列はparticipant_id,probabilityの2列にしてください。')
    if any(len(row) != 2 for row in rows[1:]):
        raise ValueError('各行をIDと確率の2項目にしてください。空行も削除してください。')
    submitted = pd.DataFrame(rows[1:], columns=rows[0])
    if len(submitted) != len(answer):
        raise ValueError(f'行数は{len(answer)}行にしてください。')
    if submitted.participant_id.duplicated().any():
        raise ValueError('IDが重複しています。1人につき1行にしてください。')
    if set(submitted.participant_id) != set(answer.participant_id):
        raise ValueError('IDがpublic testと一致しません。配布データのIDを使ってください。')
    submitted['probability'] = pd.to_numeric(submitted.probability, errors='coerce')
    if not np.isfinite(submitted.probability).all():
        raise ValueError('probabilityに欠損・文字・無限大があります。')
    if not submitted.probability.between(0, 1).all():
        raise ValueError('probabilityを0〜1の数値にしてください。')
    return submitted.set_index('participant_id').loc[answer.participant_id, 'probability']


uploaded = st.file_uploader('submission.csvをアップロード', type=['csv'], max_upload_size=1)
if uploaded is not None:
    try:
        probability = read_submission(uploaded.getvalue())
    except ValueError as error:
        st.error(str(error))
        st.stop()
    except Exception:
        st.error('読み込めませんでした。CSVを確認してください。')
        st.stop()
    y = answer.target
    auc = roc_auc_score(y, probability)
    st.metric('ROC-AUC', f'{auc:.4f}')
    if auc >= PASS_AUC:
        st.success('合格：合格ラインをクリアしました。')
    else:
        st.warning('再挑戦：validationでモデルや設定を見直してみましょう。')
    if auc >= 0.79:
        st.success('挑戦目標（0.79）も達成しました。')
    result = pd.DataFrame([{
        '課題': 'NHANES 医療AI演習',
        '採点日時（日本時間）': datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
        'ROC-AUC': auc,
        '合格ライン': PASS_AUC,
        '判定': '合格' if auc >= PASS_AUC else '再挑戦',
        '対象人数': len(probability)
    }])
    st.download_button('採点結果をダウンロード',
        data=result.to_csv(index=False).encode('utf-8-sig'),
        file_name='採点結果.csv', mime='text/csv', on_click='ignore')
    st.write('ダウンロードした採点結果.csvをLMSの課題へ提出してください。レポートは不要です。')
    prediction = probability >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    st.caption('以下の補助指標は、確率0.5以上を1として計算しています。')
    st.dataframe(pd.DataFrame([{
        'Accuracy': accuracy_score(y, prediction),
        'Sensitivity / Recall': recall_score(y, prediction, zero_division=0),
        'Specificity': tn / (tn + fp),
        'Precision': precision_score(y, prediction, zero_division=0),
        'F1': f1_score(y, prediction, zero_division=0)
    }]).round(4), hide_index=True)
    with st.expander('混同行列・ROC曲線を見る'):
        st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]],
            index=['実際0', '実際1'], columns=['予測0', '予測1']))
        fpr, tpr, _ = roc_curve(y, probability)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f'AUC={auc:.3f}')
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set(xlabel='False positive rate', ylabel='Sensitivity', xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

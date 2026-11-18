"""学生がCSVをアップロードすると即時に採点する。答案はサーバーだけに置く。"""
from pathlib import Path
import csv
import os
import io
import hashlib
import re
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
    precision_score, f1_score, confusion_matrix, roc_curve)

from ranking import RankingStore

DEFAULT_ASSIGNMENT = 'nhanes_ml_v2_2026'
COURSE_ALIASES = {'nhanes_hba1c_2026': DEFAULT_ASSIGNMENT, 'blood_dl_2026': 'blood_mlp_v2_2026'}
COURSES = {
    'nhanes_logistic_2026': {'title': 'ロジスティック回帰演習', 'pass_auc': 0.77, 'challenge_auc': 0.79,
                            'label': '01 回帰・ロジスティック回帰'},
    DEFAULT_ASSIGNMENT: {'title': '機械学習演習（HbA1c高値群）', 'pass_auc': 0.73, 'challenge_auc': None,
                         'label': '02 機械学習（SVM・決定木・ランダムフォレスト）'},
    'blood_mlp_v2_2026': {'title': '深層学習演習（単球と未熟顆粒球）', 'pass_auc': 0.82, 'challenge_auc': None,
                         'label': '03 深層学習（血球画像）'},
}
st.set_page_config(page_title='演習・提出CSVの採点', layout='centered')
try:
    assignment = dict(st.secrets.get('assignment', {}))
except FileNotFoundError:
    assignment = {}
if not assignment:
    course_ids = list(COURSES)
    requested = st.query_params.get('course', DEFAULT_ASSIGNMENT)
    requested = COURSE_ALIASES.get(requested, requested)
    selected = st.selectbox('演習を選択', course_ids,
        index=course_ids.index(requested) if requested in COURSES else course_ids.index(DEFAULT_ASSIGNMENT),
        format_func=lambda value: COURSES[value]['label'])
    st.query_params['course'] = selected
    assignment = dict(COURSES[selected], id=selected)
try:
    ASSIGNMENT_ID = assignment.get('id', DEFAULT_ASSIGNMENT)
    ASSIGNMENT_TITLE = assignment.get('title', 'NHANES 医療AI演習')
    PASS_AUC = float(assignment.get('pass_auc', 0.74))
    CHALLENGE_AUC = assignment.get('challenge_auc', 0.79 if ASSIGNMENT_ID == DEFAULT_ASSIGNMENT else None)
    if CHALLENGE_AUC is not None:
        CHALLENGE_AUC = float(CHALLENGE_AUC)
    if (not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,47}', ASSIGNMENT_ID)
            or not isinstance(ASSIGNMENT_TITLE, str) or not ASSIGNMENT_TITLE.strip()
            or not np.isfinite(PASS_AUC) or not 0 <= PASS_AUC <= 1
            or (CHALLENGE_AUC is not None and (not np.isfinite(CHALLENGE_AUC) or not PASS_AUC <= CHALLENGE_AUC <= 1))):
        raise ValueError()
except (TypeError, ValueError):
    st.error('課題の設定を確認してください。')
    st.stop()
ANSWER_FILE = Path(os.environ.get('NHANES_ANSWER_FILE', str(Path(__file__).with_name('public_test_answer.csv'))))

PAGE_TITLE = f'{ASSIGNMENT_TITLE}・提出CSVの採点'
st.title(PAGE_TITLE)
st.write('submission.csvを選ぶと、自動で点数と合否が表示されます。結果をダウンロードしてLMSへ提出してください。')
st.caption(f'合格：ROC-AUC {PASS_AUC:g}以上。提出の時間・期間・回数に制限はありません。')

try:
    local_answers = os.environ.get('NHANES_ANSWERS_DIR')
    if ASSIGNMENT_ID == 'nhanes_hba1c_2026':
        source = ANSWER_FILE if ANSWER_FILE.exists() else io.StringIO(st.secrets['answer_csv'])
    elif local_answers and (Path(local_answers) / f'{ASSIGNMENT_ID}.csv').is_file():
        source = Path(local_answers) / f'{ASSIGNMENT_ID}.csv'
    elif ASSIGNMENT_ID in st.secrets.get('answers', {}):
        source = io.StringIO(st.secrets['answers'][ASSIGNMENT_ID])
    else:
        raise ValueError('Unknown assignment')
    answer = pd.read_csv(source, dtype={'participant_id': str})
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
    if rows and 'student_id' in rows[0]:
        raise ValueError('旧形式のCSVです。最新版の配布ノートで学籍番号を含まないCSVを作成してください。')
    if not rows or rows[0] != ['participant_id', 'probability']:
        raise ValueError('列はparticipant_id,probabilityの2列にしてください。')
    if any(len(row) != 2 for row in rows[1:]):
        raise ValueError('各行を対象ID・予測値の2項目にしてください。空行も削除してください。')
    submitted = pd.DataFrame(rows[1:], columns=rows[0])
    if len(submitted) != len(answer):
        raise ValueError(f'行数は{len(answer)}行にしてください。')
    if submitted.participant_id.duplicated().any():
        raise ValueError('IDが重複しています。対象IDごとに1行にしてください。')
    if set(submitted.participant_id) != set(answer.participant_id):
        raise ValueError('IDがpublic testと一致しません。配布データのIDを使ってください。')
    submitted['probability'] = pd.to_numeric(submitted.probability, errors='coerce')
    if not np.isfinite(submitted.probability).all():
        raise ValueError('probabilityに欠損・文字・無限大があります。')
    if not submitted.probability.between(0, 1).all():
        raise ValueError('probabilityを0〜1の数値にしてください。')
    return submitted.set_index('participant_id').loc[answer.participant_id, 'probability']



def ranking_settings():
    try:
        settings = dict(st.secrets.get('leaderboard', {}))
    except FileNotFoundError:
        return {}
    return settings


@st.cache_resource
def ranking_store(url, api_token, assignment_id):
    # 旧形式の「学生ごとの最高点」と匿名の提出を別のシートに保管する。
    board_id = 'anon_' + hashlib.sha256(assignment_id.encode()).hexdigest()[:40]
    return RankingStore(url, api_token, assignment_id=board_id)


@st.fragment(run_every=30)
def show_ranking(store, upload_key, auc=None):
    st.subheader('この課題の上位20スコア')
    st.caption('提出ごとのROC-AUCを高い順に表示します。同じ人の複数回の挑戦を含みます。約30秒ごとに更新します。成績はWebClassへの提出後に確定します。')
    st.button('ランキングを更新')
    try:
        scores = store.snapshot()
        saved = upload_key is not None and auc is not None and dict(scores).get(upload_key, -1) >= auc
        ticket = st.session_state.get('_ranking_ticket')
        if upload_key and ticket is not None and not saved:
            if not ticket.done():
                st.info('ランキングへ登録中です。反映まで少しお待ちください。')
            elif ticket.exception() is not None:
                st.warning('ランキングへ保存できませんでした。採点結果はダウンロードできます。')
                if st.button('ランキング登録を再試行'):
                    st.session_state['_ranking_ticket'] = store.submit(upload_key, auc)
                    st.rerun(scope='fragment')
        values = sorted((value for _, value in scores), reverse=True)[:20]
        if not values:
            st.info('まだ登録されたスコアがありません。')
            return
        st.dataframe(pd.DataFrame({'ROC-AUC': values}), hide_index=True,
                     column_config={'ROC-AUC': st.column_config.NumberColumn(format='%.6f')})
    except Exception:
        st.warning('ランキングを読み込めませんでした。少し待って更新してください。')


st.caption('予測CSVには対象ID（participant_id）と予測値（probability）の2列を入れてください。氏名・学籍番号は不要です。')
uploaded = st.file_uploader('submission.csvをアップロード', type=['csv'], max_upload_size=1, key=f'csv_{ASSIGNMENT_ID}')
settings = ranking_settings()
upload_key = None
auc = None
store = None
if settings:
    try:
        store = ranking_store(settings['url'], settings['api_token'], ASSIGNMENT_ID)
    except Exception:
        st.warning('ランキングの準備ができていません。採点は利用できます。')
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
    if settings:
        try:
            # Streamlit issues a fresh random file_id per upload and retains it on reruns.
            upload_key = hashlib.sha256(uploaded.file_id.encode()).hexdigest()
            attempt = (ASSIGNMENT_ID, upload_key)
            if store is not None and st.session_state.get('_ranking_upload') != attempt:
                st.session_state['_ranking_ticket'] = store.submit(upload_key, float(auc))
                st.session_state['_ranking_upload'] = attempt
        except Exception:
            st.warning('採点は完了しましたが、ランキングへ保存できませんでした。少し待って再度アップロードしてください。')
    if auc >= PASS_AUC:
        st.success('合格：合格ラインをクリアしました。')
    else:
        st.warning('再挑戦：validationでモデルや設定を見直してみましょう。')
    if CHALLENGE_AUC is not None and auc >= CHALLENGE_AUC:
        st.success(f'挑戦目標（{CHALLENGE_AUC:g}）も達成しました。')
    result = pd.DataFrame([{
        '課題': ASSIGNMENT_TITLE,
        '課題ID': ASSIGNMENT_ID,
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

if store is not None:
    show_ranking(store, upload_key, auc)

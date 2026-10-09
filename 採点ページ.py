"""学生がCSVをアップロードすると即時に採点する。答案はサーバーだけに置く。"""
from pathlib import Path
import csv
import os
import io
import hmac
import hashlib
import re
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
    precision_score, f1_score, confusion_matrix, roc_curve)

from ranking import RankingStore, ranking_rows

DEFAULT_ASSIGNMENT = 'nhanes_hba1c_2026'
COURSES = {
    'nhanes_logistic_2026': {'title': 'ロジスティック回帰コンペ', 'pass_auc': 0.74, 'challenge_auc': 0.79,
                            'label': '01 回帰・ロジスティック回帰'},
    DEFAULT_ASSIGNMENT: {'title': 'NHANES 医療AI演習', 'pass_auc': 0.74, 'challenge_auc': 0.79,
                         'label': '02 機械学習（SVM・決定木・ランダムフォレスト）'},
    'breast_dl_2026': {'title': '医療画像・深層学習コンペ', 'pass_auc': 0.80, 'challenge_auc': 0.85,
                         'label': '03 深層学習（医療画像）'},
}
st.set_page_config(page_title='コンペ・提出CSVの採点', layout='centered')
try:
    assignment = dict(st.secrets.get('assignment', {}))
except FileNotFoundError:
    assignment = {}
if not assignment:
    course_ids = list(COURSES)
    requested = st.query_params.get('course', DEFAULT_ASSIGNMENT)
    selected = st.selectbox('コンペを選択', course_ids,
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

PAGE_TITLE = '医療AI演習・提出CSVの採点' if ASSIGNMENT_ID == DEFAULT_ASSIGNMENT else f'{ASSIGNMENT_TITLE}・提出CSVの採点'
st.title(PAGE_TITLE)
st.write('submission.csvを選ぶと、自動で点数と合否が表示されます。結果をダウンロードしてLMSへ提出してください。')
st.caption(f'合格：ROC-AUC {PASS_AUC:g}以上。提出の時間・期間・回数に制限はありません。')

try:
    if ASSIGNMENT_ID != DEFAULT_ASSIGNMENT:
        local_answers = os.environ.get('NHANES_ANSWERS_DIR')
        if local_answers:
            answer = pd.read_csv(Path(local_answers) / f'{ASSIGNMENT_ID}.csv', dtype={'participant_id': str})
        else:
            answer = pd.read_csv(io.StringIO(st.secrets['answers'][ASSIGNMENT_ID]), dtype={'participant_id': str})
    elif ANSWER_FILE.exists():
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
    if not rows or rows[0] != ['student_id', 'participant_id', 'probability']:
        raise ValueError('列はstudent_id,participant_id,probabilityの3列にしてください。')
    if any(len(row) != 3 for row in rows[1:]):
        raise ValueError('各行を学籍番号・対象ID・予測値の3項目にしてください。空行も削除してください。')
    submitted = pd.DataFrame(rows[1:], columns=rows[0])
    if len(submitted) != len(answer):
        raise ValueError(f'行数は{len(answer)}行にしてください。')
    student_ids = submitted.student_id.str.strip().str.upper()
    if not student_ids.str.fullmatch(r'[A-Z0-9][A-Z0-9-]{0,31}').all() or student_ids.nunique() != 1:
        raise ValueError('student_idは全行に同じ学籍番号を入れてください。半角英数字・ハイフンの32文字以内です。')
    if submitted.participant_id.duplicated().any():
        raise ValueError('IDが重複しています。1人につき1行にしてください。')
    if set(submitted.participant_id) != set(answer.participant_id):
        raise ValueError('IDがpublic testと一致しません。配布データのIDを使ってください。')
    submitted['probability'] = pd.to_numeric(submitted.probability, errors='coerce')
    if not np.isfinite(submitted.probability).all():
        raise ValueError('probabilityに欠損・文字・無限大があります。')
    if not submitted.probability.between(0, 1).all():
        raise ValueError('probabilityを0〜1の数値にしてください。')
    return student_ids.iloc[0], submitted.set_index('participant_id').loc[answer.participant_id, 'probability']



def ranking_settings():
    try:
        settings = dict(st.secrets.get('leaderboard', {}))
    except FileNotFoundError:
        return {}
    return settings


def student_key(student_id, secret):
    if len(secret) < 32:
        raise ValueError('ランキングの設定を確認してください。')
    return hmac.new(secret.encode(), student_id.encode(), hashlib.sha256).hexdigest()


@st.cache_resource
def ranking_store(url, api_token, assignment_id):
    return RankingStore(url, api_token, assignment_id=assignment_id)


def public_ranking(rows):
    """ブラウザへ渡す列を限定する。学籍番号や内部キーは含めない。"""
    return pd.DataFrame([{'順位': int(row['position']), 'ROC-AUC': float(row['auc']),
                          'あなた': '← あなた' if row['is_self'] else ''} for row in rows])


@st.fragment(run_every=30)
def show_ranking(store, own_key, auc=None):
    st.subheader('現在のランキング')
    st.caption('1人につき最高スコアを掲載します。同点は同順位です。更新は約30秒ごとです。成績はLMSへの提出後に確定します。')
    st.button('ランキングを更新')
    try:
        scores = store.snapshot()
        saved = own_key is not None and auc is not None and dict(scores).get(own_key, -1) >= auc
        ticket = st.session_state.get('_ranking_ticket')
        if own_key and ticket is not None and not saved:
            if not ticket.done():
                st.info('ランキングへ登録中です。反映まで少しお待ちください。')
            elif ticket.exception() is not None:
                st.warning('ランキングへ保存できませんでした。採点結果はダウンロードできます。')
                if st.button('ランキング登録を再試行'):
                    st.session_state['_ranking_ticket'] = store.submit(own_key, auc)
                    st.rerun(scope='fragment')
        rows = ranking_rows(scores, own_key)
        if not rows:
            st.info('まだ登録されたスコアがありません。')
            return
        own = next((row for row in rows if row['is_self']), None)
        if own:
            st.write(f"あなたは現在 **{own['position']}位 / {own['participants']}人** です。最高ROC-AUC：**{own['auc']:.6f}**")
        elif not own_key:
            st.caption('自分の順位はsubmission.csvをアップロードすると表示されます。')
        st.dataframe(public_ranking(rows), hide_index=True,
                     column_config={'ROC-AUC': st.column_config.NumberColumn(format='%.6f')})
    except Exception:
        st.warning('ランキングを読み込めませんでした。少し待って更新してください。')


uploaded = st.file_uploader('submission.csvをアップロード', type=['csv'], max_upload_size=1, key=f'csv_{ASSIGNMENT_ID}')
settings = ranking_settings()
own_key = None
auc = None
store = None
if settings:
    try:
        store = ranking_store(settings['url'], settings['api_token'], ASSIGNMENT_ID)
    except Exception:
        st.warning('ランキングの準備ができていません。採点は利用できます。')
if uploaded is not None:
    try:
        student_id, probability = read_submission(uploaded.getvalue())
    except ValueError as error:
        st.error(str(error))
        st.stop()
    except Exception:
        st.error('読み込めませんでした。CSVを確認してください。')
        st.stop()
    y = answer.target
    auc = roc_auc_score(y, probability)
    st.write(f'学籍番号：{student_id}（ランキングには表示されません）')
    st.metric('ROC-AUC', f'{auc:.4f}')
    if settings:
        try:
            key_id = student_id if ASSIGNMENT_ID == DEFAULT_ASSIGNMENT else f'{ASSIGNMENT_ID}:{student_id}'
            own_key = student_key(key_id, settings['identity_secret'])
            fingerprint = hashlib.sha256(ASSIGNMENT_ID.encode() + b'\0' + uploaded.getvalue()).hexdigest()
            if store is not None and st.session_state.get('_ranking_file') != fingerprint:
                st.session_state['_ranking_ticket'] = store.submit(own_key, float(auc))
                st.session_state['_ranking_file'] = fingerprint
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
        '学籍番号': student_id,
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
    show_ranking(store, own_key, auc)

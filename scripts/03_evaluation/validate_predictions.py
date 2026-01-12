"""
예측 결과 검증 스크립트
제출 전에 반드시 실행하여 형식 확인
"""

import pandas as pd
import numpy as np
import os

def validate_predictions():
    """예측 파일 형식 검증"""

    print("=" * 60)
    print("예측 결과 검증")
    print("=" * 60)

    # 파일 경로
    pred_file = 'pred_subtask2a.csv'

    # 1. 파일 존재 확인
    if not os.path.exists(pred_file):
        print(f"❌ 파일이 없습니다: {pred_file}")
        print(f"   예측 스크립트를 먼저 실행하세요.")
        return False

    print(f"✅ 파일 발견: {pred_file}")

    # 2. 파일 로드
    try:
        pred_df = pd.read_csv(pred_file)
        print(f"✅ 파일 로드 성공: {len(pred_df)} rows")
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return False

    # 3. 컬럼 확인 (정확한 순서와 이름)
    expected_columns = ['user_id', 'pred_state_change_valence', 'pred_state_change_arousal']
    actual_columns = pred_df.columns.tolist()

    print(f"\n컬럼 확인:")
    print(f"  예상: {expected_columns}")
    print(f"  실제: {actual_columns}")

    if actual_columns != expected_columns:
        print(f"❌ 컬럼이 정확히 일치하지 않습니다!")
        print(f"   순서와 이름을 정확히 맞춰야 합니다.")
        return False
    else:
        print(f"✅ 컬럼 형식 정확함")

    # 4. 데이터 타입 확인
    print(f"\n데이터 타입 확인:")
    print(pred_df.dtypes)

    # user_id는 int, 나머지는 float
    if not pd.api.types.is_integer_dtype(pred_df['user_id']):
        print(f"⚠️ user_id가 정수가 아닙니다")

    if not pd.api.types.is_float_dtype(pred_df['pred_state_change_valence']):
        print(f"⚠️ pred_state_change_valence가 float가 아닙니다")

    if not pd.api.types.is_float_dtype(pred_df['pred_state_change_arousal']):
        print(f"⚠️ pred_state_change_arousal가 float가 아닙니다")

    # 5. NaN/Null 확인
    nan_count = pred_df.isna().sum().sum()
    print(f"\nNaN/Null 확인:")
    if nan_count > 0:
        print(f"❌ NaN 값 발견: {nan_count}개")
        print(pred_df.isna().sum())
        return False
    else:
        print(f"✅ NaN 없음")

    # 6. 중복 user_id 확인
    duplicates = pred_df[pred_df.duplicated('user_id', keep=False)]
    if len(duplicates) > 0:
        print(f"\n❌ 중복된 user_id 발견: {len(duplicates)}개")
        print(duplicates)
        return False
    else:
        print(f"✅ 중복 user_id 없음")

    # 7. 값 범위 확인
    print(f"\n값 범위 확인:")
    print(pred_df.describe())

    # 일반적인 범위 체크 (경고만)
    val_min, val_max = pred_df['pred_state_change_valence'].min(), pred_df['pred_state_change_valence'].max()
    aro_min, aro_max = pred_df['pred_state_change_arousal'].min(), pred_df['pred_state_change_arousal'].max()

    print(f"\nValence 범위: [{val_min:.3f}, {val_max:.3f}]")
    print(f"Arousal 범위: [{aro_min:.3f}, {aro_max:.3f}]")

    if abs(val_min) > 10 or abs(val_max) > 10:
        print(f"⚠️ Valence 값이 비정상적으로 큽니다 (일반적으로 -4 ~ +4)")

    if abs(aro_min) > 10 or abs(aro_max) > 10:
        print(f"⚠️ Arousal 값이 비정상적으로 큽니다 (일반적으로 -2 ~ +2)")

    # 8. 테스트 파일과 비교
    test_file = 'data/test/test_subtask2a.csv'
    if os.path.exists(test_file):
        test_df = pd.read_csv(test_file)

        if 'is_forecasting_user' in test_df.columns:
            forecasting_users = test_df[test_df['is_forecasting_user'] == True]['user_id'].unique()

            print(f"\n사용자 수 확인:")
            print(f"  Test forecasting users: {len(forecasting_users)}명")
            print(f"  Prediction users: {len(pred_df)}명")

            if len(pred_df) != len(forecasting_users):
                print(f"⚠️ 예측 사용자 수가 일치하지 않습니다")

            # 누락된 사용자 확인
            pred_users = set(pred_df['user_id'].unique())
            expected_users = set(forecasting_users)

            missing = expected_users - pred_users
            extra = pred_users - expected_users

            if missing:
                print(f"❌ 예측에서 누락된 사용자: {missing}")
                return False

            if extra:
                print(f"⚠️ 예측에 추가된 사용자 (예측 대상 아님): {extra}")

    # 9. 데이터 미리보기
    print(f"\n데이터 미리보기 (처음 10개):")
    print(pred_df.head(10))

    print("\n" + "=" * 60)
    print("✅ 검증 완료 - 제출 가능합니다!")
    print("=" * 60)

    return True

if __name__ == "__main__":
    result = validate_predictions()

    if result:
        print("\n다음 단계:")
        print("1. submission.zip 생성:")
        print("   powershell Compress-Archive -Path pred_subtask2a.csv -DestinationPath submission.zip -Force")
        print("\n2. Codabench에 제출:")
        print("   https://www.codabench.org/competitions/9963/")
    else:
        print("\n⚠️ 문제를 해결한 후 다시 검증하세요.")

"""
평가 데이터 검증 스크립트
평가파일을 받은 후 즉시 실행하여 형식 확인
"""

import pandas as pd
import os

def verify_test_data():
    """평가 데이터 형식 검증"""

    print("=" * 60)
    print("평가 데이터 검증 시작")
    print("=" * 60)

    # 파일 경로
    test_file = 'data/test/test_subtask2a.csv'

    # 1. 파일 존재 확인
    if not os.path.exists(test_file):
        print(f"❌ 파일이 없습니다: {test_file}")
        print(f"   다운로드한 파일을 {test_file} 경로에 저장하세요.")
        return False

    print(f"✅ 파일 발견: {test_file}")

    # 2. 파일 로드
    try:
        test_df = pd.read_csv(test_file)
        print(f"✅ 파일 로드 성공: {len(test_df)} rows")
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return False

    # 3. 컬럼 확인
    print(f"\n컬럼 목록: {test_df.columns.tolist()}")

    expected_columns = ['user_id', 'is_forecasting_user']
    missing_cols = set(expected_columns) - set(test_df.columns)

    if missing_cols:
        print(f"⚠️ 예상 컬럼 누락: {missing_cols}")
    else:
        print(f"✅ 필수 컬럼 확인됨: {expected_columns}")

    # 4. Forecasting users 확인
    if 'is_forecasting_user' in test_df.columns:
        forecasting_users = test_df[test_df['is_forecasting_user'] == True]
        print(f"\n✅ Forecasting users: {len(forecasting_users)}명")
        print(f"   User IDs: {sorted(forecasting_users['user_id'].tolist())}")
    else:
        print(f"\n⚠️ 'is_forecasting_user' 컬럼이 없습니다")

    # 5. Training data와 비교
    train_file = 'data/raw/train_subtask2a.csv'

    if os.path.exists(train_file):
        train_df = pd.read_csv(train_file)
        train_users = set(train_df['user_id'].unique())

        if 'user_id' in test_df.columns:
            test_users = set(test_df['user_id'].unique())

            # 겹치는 사용자 확인
            overlap = test_users & train_users
            new_users = test_users - train_users

            print(f"\n사용자 비교:")
            print(f"  Training users: {len(train_users)}명")
            print(f"  Test users: {len(test_users)}명")
            print(f"  Overlap: {len(overlap)}명")

            if new_users:
                print(f"  ⚠️ 새로운 사용자: {len(new_users)}명")
                print(f"     {sorted(list(new_users))[:10]}...")  # 처음 10명만
            else:
                print(f"  ✅ 모든 test users가 training에 존재")

    # 6. 데이터 미리보기
    print(f"\n데이터 미리보기:")
    print(test_df.head(10))

    # 7. 통계
    print(f"\n데이터 통계:")
    print(test_df.describe())

    print("\n" + "=" * 60)
    print("✅ 검증 완료")
    print("=" * 60)

    return True

if __name__ == "__main__":
    verify_test_data()

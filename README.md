이 코드는 비트코인 자동 거래 시스템을 구현한 것입니다. 주요 기능:

1. `TradingDatabase` 클래스
- SQLite DB로 거래 내역 관리
- 거래 성과 분석 및 반성(reflection) 기록
- 최근 거래 내역 조회

2. `BithumbAPI` 클래스
- 빗썸 API 연동
- 계좌 정보 조회
- 현재 가격 조회
- 잔액 정보 출력 및 DB 기록

3. 주요 분석 기능
- 기술적 지표(RSI, MACD, 볼린저밴드 등) 계산
- 공포탐욕지수 분석
- 비트코인 뉴스 수집
- 차트 이미지 캡처

4. AI 거래 의사결정
- GPT-4 활용
- 기술적 지표, 뉴스, 차트 분석
- 매수/매도/홀드 결정
- 리스크 레벨 및 신뢰도 평가
- 거래 파라미터 제안

5. 거래 실행
- 설정된 포지션 크기로 시장가 주문
- 거래 후 성과 분석
- 거래 전략 개선을 위한 반성 기록

이 시스템은 데이터 분석, AI 의사결정, 자동화된 거래 실행을 통합하여 체계적인 암호화폐 트레이딩을 구현합니다.
=======================================================================================

이 코드는 Streamlit으로 구현된 비트코인 거래 대시보드입니다. 주요 기능:

1. 데이터 시각화
- 포트폴리오 가치 추이
- BTC 가격 히스토리
- 트레이딩 활동 지표

2. 거래 분석
- 총 거래 수
- 매수/매도 주문 수
- 최근 거래 내역 테이블

3. 성과 분석
- 수익률 지표
- 시장 상황 분석
- 학습된 교훈
- 전략 조정사항

4. 필터링 기능
- 기간 선택 슬라이더
- 날짜 기반 데이터 필터링

SQLite DB에서 거래 내역과 분석 데이터를 불러와 Plotly로 시각화하고 Streamlit으로 대시보드를 구성합니다.

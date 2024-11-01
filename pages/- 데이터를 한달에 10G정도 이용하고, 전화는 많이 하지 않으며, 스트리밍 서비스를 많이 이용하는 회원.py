
import streamlit as st
st.set_page_config(layout='wide')

st.title("- 데이터를 한달에 10G정도 이용하고, 전화는 많이 하지 않으며, 스트리밍 서비스를 많이 이용하는 회원")

# Displaying the value of A on this new page
st.markdown("""데이터를 한달에 10G정도 이용하고, 전화는 많이 하지 않으며, 스트리밍 서비스를 많이 이용하는 회원""")

col1, col2 = st.columns(2, gap='large')

with col1:
    st.header("고객 특성")
    st.markdown("""#### A : 프로파일 정보 요약
- 데이터를 한달에 10G정도 이용하고, 전화는 많이 하지 않으며, 스트리밍 서비스를 많이 이용하는 회원

#### B : 고객군 특성

##### 1. 프로파일 요약
이 고객은 한 달에 약 10GB의 데이터를 사용하며, 전화 통화는 많이 하지 않습니다. 주로 스트리밍 서비스를 많이 이용하는 특징이 있습니다.

##### 2. 주요 심리적 동인
이 고객의 주요 심리적 동인은 경제적 효율성과 소비자의 주권입니다. 즉, 통신비를 절감하면서도 자신에게 맞는 최적의 요금제를 찾고자 하는 의지가 강합니다.

##### 3. 주요 핵심 가치
이 고객은 비용 절감과 유연성을 중요시합니다. 큰 통신사에 의존하기보다는 자신에게 맞는 가성비 높은 서비스를 선택하려는 경향이 있습니다.

##### 4. 주요 관심사
이 고객은 데이터 사용량에 맞는 요금제와 스트리밍 서비스 혜택에 관심이 많습니다. 또한, 통화와 문자 사용이 적기 때문에 이에 맞는 저렴한 요금제를 선호합니다.
""")

with col2:
    st.header("카테고리 추천")
    st.markdown("""

##### 1. 5G 시니어 B형
- 데이터 이용량: 데이터 10GB + 다 쓰면 최대 1Mbps
- 요금제: 월 43,000원 (약정 할인: 32,250원)
- 혜택: 집/이동전화 무제한, 부가통화 400분, 문자메시지 기본제공, U⁺ 모바일tv 라이트 무료

이 요금제는 데이터 10GB를 제공하며, 다 사용한 후에도 최대 1Mbps 속도로 계속 사용할 수 있어 스트리밍 서비스 이용에 적합합니다. 또한, 통화와 문자 혜택도 충분히 제공됩니다.

##### 2. 5G 슬림+
- 데이터 이용량: 데이터 9GB + 다 쓰면 최대 400kbps
- 요금제: 월 47,000원 (약정 할인: 35,250원)
- 혜택: 집/이동전화 무제한, 부가통화 300분, 문자메시지 기본제공, U⁺ 모바일tv 라이트 무료

이 요금제는 데이터 9GB를 제공하며, 다 사용한 후에도 최대 400kbps 속도로 계속 사용할 수 있습니다. 스트리밍 서비스 이용에 적합하며, 통화와 문자 혜택도 충분히 제공됩니다.

##### 3. 5G 라이트+
- 데이터 이용량: 데이터 14GB + 다 쓰면 최대 1Mbps
- 요금제: 월 55,000원 (약정 할인: 41,250원)
- 혜택: 집/이동전화 무제한, 부가통화 300분, 문자메시지 기본제공, U⁺ 모바일tv 라이트 무료

이 요금제는 데이터 14GB를 제공하며, 다 사용한 후에도 최대 1Mbps 속도로 계속 사용할 수 있어 스트리밍 서비스 이용에 매우 적합합니다. 또한, 통화와 문자 혜택도 충분히 제공됩니다.""")

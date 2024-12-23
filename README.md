# MMaction2_TSN_Lightweight_CNN_Architecture


[HUFS_DeapLearning_Team2_Notion](https://j8n17.notion.site/HUFS-DL-2-f94647f9b8d642a9bce8c5533282c3e3?pvs=4)

[정보과학회논문지(JOK)](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003136950)

Lightweight Temporal Segment Network for Video Scene
Understanding: Validation in Driver Assault Detection


요 약 

최근 택시와 버스 등 교통수단에서 탑승자가 운전자를 폭행하는 사건이 증가하는 추세로, 특히
늦은 밤 주취자에 의한 운전자 폭행 등에 대한 신속한 대응은 더욱 어려운 상황이다. 이러한 문제에 대응
하기 위해, 본 연구팀은 탑승자에 의한 운전자 폭행 상황을 실시간으로 탐지할 수 있는 경량 합성곱 신경
망 기반의 시간적 세그먼트 네트워크(TSN) 모델을 제안한다. TSN은 동영상을 효율적으로 처리하기 위해
소수의 이미지 프레임을 샘플링하며, 공간 정보처리 스트림과 시간 정보처리 스트림으로 나뉘어 학습이 진
행된다. 각 스트림에는 합성곱 신경망이 들어가는데, 이 연구에서는 경량 신경망 아키텍처인 MobileOne
모델을 적용하여 모델 사이즈를 크게 줄였고, 제한된 컴퓨팅 리소스에서도 정확도는 오히려 개선됨을 보인
다. 본 모델은 차량 내 운전자 모니터링 시스템에 통합되어 운전자에게 발생할 수 있는 위험한 상황에 대
한 신속한 대응 및 예방에 기여할 수 있을 것으로 기대된다.


키워드: 

시간 세그먼트 네트워크, 비디오 장면 이해, 이상 탐지, 운전자 폭행


Abstract 

The number of driver assaults in transportation such as taxis and buses has been
increasing over the past few years. It can be especially difficult to respond quickly to assaults on
drivers by drunks late at night. To address this issue, our research team proposed a lightweight
CNN-based Temporal Segment Network (TSN) model that could detect driver assaults by passengers
in real time. The TSN model efficiently processes videos by sampling a small number of image frames
and divides videos into two streams for learning: one for spatial information processing and the other
for temporal information processing. Convolutional neural networks are employed in each stream. In
this research, we applied a lightweight CNN architecture, MobileOne, significantly reducing the model
size while demonstrating improved accuracy even with limited computing resources. The model is
expected to contribute to rapid response and prevention of hazardous situations for drivers when it is
integrated into vehicular driver monitoring systems.


Keywords:

temporal segment network, video scene understanding, anomaly detection, driver
assault

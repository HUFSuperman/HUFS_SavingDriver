# HUFS_SavingDriver

[HUFS_DeapLearning_Team2_Notion](https://j8n17.notion.site/HUFS-DL-2-f94647f9b8d642a9bce8c5533282c3e3?pvs=4)

Lightweight CNN Architecture-Based Temporal Segment Network Technique for Driver Assault Detection


최근, 운전 중 발생하는 폭행 사건이 증가하고 있으며, 특히 주취자에 의한 폭행은 주로 늦은 시간에 발생해 신속한 대응이 어려운 상황입니다. 이러한 문제에 대응하기 위해, 본 연구는 운전자 이외의 탑승자에 의한 운전자 폭행 상황을 신속하고 정확하게 탐지할 수 있는 경량 컨볼루션 신경망(CNN) 기반의 시간적 세그먼트 네트워크(TSN) 모델을 개발하였습니다. 이 모델은 차량 내 운전자 모니터링 시스템(DMS)에 통합되어 제한된 자원 하에서도 효율적으로 작동하도록 설계되었습니다. 기존의 ResNet 기반 TSN 모델을 개선하여, 자원 제약 조건 하에서도 높은 성능을 유지할 수 있는 경량 CNN 아키텍처를 도입하였습니다. 본 연구에서 제안하는 기법은 운전 중 발생할 수 있는 위험한 상황에 대한 신속한 대응 및 예방에 기여할 것으로 기대됩니다.




데이터 사용 설명서 : convert -> assault_split -> EDA 순으로 실행


- convert.ipynb : json을 train.csv로 변환

- assault_split.ipynb : train.csv에서 폭행 데이터만 추출

- EDA.ipynb : EDA


운전자 폭행 탐지를 위한 TSN 경량 CNN 아키텍처 도입 

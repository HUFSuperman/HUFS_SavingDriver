# HUFS_SavingDriver

[HUFS_DeapLearning_Team2_Notion](https://j8n17.notion.site/HUFS-DL-2-f94647f9b8d642a9bce8c5533282c3e3?pvs=4)

Lightweight CNN Architecture-Based Temporal Segment Network Technique for Driver Assault Detection

최근 운전 중 발생하는 폭행 사건이 증가하고 있으며, 특히 주취자에 의한 폭행은 주로 늦은 시간에 발생해 신속한 대응이 어려운 상황이다. 이러한 문제에 대응하기 위해, 본 연구는 운전자 이외의 탑승자에 의한 운전자 폭행 상황을 신속하고 정확하게 탐지할 수 있는 경량 컨볼루션 신경망(CNN) 기반의 시간적 세그먼트 네트워크(TSN) 모델을 제안하고자 한다. 이 모델은 차량 내 운전자 모니터링 시스템(DMS)에 통합되어 제한된 자원 하에서도 효율적으로 작동하도록 설계되었다. 기존의 ResNet 기반 TSN 모델을 개선하여, 자원 제약 조건 하에서도 높은 성능을 유지할 수 있는 경량 CNN 아키텍처를 도입하였다. 본 연구에서 제안하는 기법은 운전 중 발생할 수 있는 위험한 상황에 대한 신속한 대응 및 예방에 기여할 것으로 기대된다.

핵심어: 시간적 세그먼트 네트워크,경량 합성곱 신경망 아키텍처, 3D 비디오 RGB 기반 폭력 인식, 비정상적 행동

Recently, there has been an increase in assault incidents occurring during driving, especially those involving intoxicated individuals, which primarily occur late at night, making prompt response challenging. To address this issue, this study proposes a lightweight Convolutional Neural Network (CNN)-based Temporal Segment Network (TSN) model capable of quickly and accurately detecting assaults on drivers by passengers other than the driver. The model is designed to operate efficiently under limited resources by integrating into the Driver Monitoring System (DMS) within the vehicle. By improving upon the existing ResNet-based TSN model, a lightweight CNN architecture is introduced that maintains high performance under resource constraints. The technique proposed in this study is expected to contribute to rapid response and prevention of dangerous situations that can arise while driving.


Keywords: Temporal Segment Network, Lightweight CNN Architecture, 3D Video RGB based Assault Recognition, Abnormal Behavior 


데이터 사용 설명서 : convert -> assault_split -> EDA 순으로 실행


- convert.ipynb : json을 train.csv로 변환

- assault_split.ipynb : train.csv에서 폭행 데이터만 추출

- EDA.ipynb : EDA


운전자 폭행 탐지를 위한 TSN 경량 CNN 아키텍처 도입 

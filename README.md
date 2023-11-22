# HUFS_SavingDriver

[HUFS_DeapLearning_Team2_Notion](https://j8n17.notion.site/HUFS-DL-2-f94647f9b8d642a9bce8c5533282c3e3?pvs=4)

Lightweight CNN Architecture-Based Temporal Segment Network Technique for Driver Assault Detection


Recently, there has been an increase in assault incidents occurring during driving, especially those involving intoxicated individuals, which primarily occur late at night, making prompt response challenging. To address this issue, this study proposes a lightweight Convolutional Neural Network (CNN)-based Temporal Segment Network (TSN) model capable of quickly and accurately detecting assaults on drivers by passengers other than the driver. The model is designed to operate efficiently under limited resources by integrating into the Driver Monitoring System (DMS) within the vehicle. By improving upon the existing ResNet-based TSN model, a lightweight CNN architecture is introduced that maintains high performance under resource constraints. The technique proposed in this study is expected to contribute to rapid response and prevention of dangerous situations that can arise while driving.

핵심어: Temporal Segment Network, Lightweight CNN Architecture, 3D Video RGB based Violence Recognition, Abnormal Behavior 


데이터 사용 설명서 : convert -> assault_split -> EDA 순으로 실행


- convert.ipynb : json을 train.csv로 변환

- assault_split.ipynb : train.csv에서 폭행 데이터만 추출

- EDA.ipynb : EDA


운전자 폭행 탐지를 위한 TSN 경량 CNN 아키텍처 도입 

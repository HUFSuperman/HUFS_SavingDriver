# HUFS_SavingDriver

[HUFS_DeapLearning_Team2_Notion](https://j8n17.notion.site/HUFS-DL-2-f94647f9b8d642a9bce8c5533282c3e3?pvs=4)

Lightweight CNN Architecture-Based Temporal Segment Network Technique for Driver Assault Detection


Recently, the incidence of assaults during driving has been on the rise, with assaults by intoxicated individuals occurring predominantly at late hours, making rapid response challenging. To address this issue, our study developed a lightweight Convolutional Neural Network (CNN) based Temporal Segment Network (TSN) model capable of swiftly and accurately detecting driver assault situations by passengers other than the driver. This model is designed to operate efficiently within the constrained resources of an in-vehicle Driver Monitoring System (DMS). We improved upon the existing ResNet-based TSN model by incorporating a lightweight CNN architecture that maintains high performance under resource constraints. The technique proposed in this study is expected to contribute to rapid response and prevention of dangerous situations that can occur while driving.

핵심어: Temporal Segment Network, Lightweight CNN Architecture, 3D Video RGB based Violence Recognition, Abnormal Behavior 


데이터 사용 설명서 : convert -> assault_split -> EDA 순으로 실행


- convert.ipynb : json을 train.csv로 변환

- assault_split.ipynb : train.csv에서 폭행 데이터만 추출

- EDA.ipynb : EDA


운전자 폭행 탐지를 위한 TSN 경량 CNN 아키텍처 도입 

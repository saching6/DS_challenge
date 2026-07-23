import pandas as pd

CAPACITY = 0.05        # ops can escalate the riskiest 5% of incidents
CALIBRATION = 'sigmoid'

GRID_SEARCH_SPACE = dict(
    n_estimators=[150, 300, 600],
    learning_rate=[0.02, 0.05, 0.1],
    max_depth=[3, 4, 6, 8],
    min_child_weight=[1, 5, 10, 20],
    scale_pos_weight=[1, 10, 35],
    reg_lambda=[1, 5, 20],
)

DEFAULT_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                      min_child_weight=10, scale_pos_weight=35, reg_lambda=5)

INCIDENTS_PATH = 'data/metro_incidents_en.csv'
MILEAGE_PATH = 'data/planned_mileage_en.csv'

MAJOR_BANDS = {'20 to 24 min', '25 to 29 min', '30 min and over'}
SINGLE_LINES = {'green', 'orange', 'blue', 'yellow'}

MILEAGE_END = pd.Timestamp('2023-09-30')   # feed stops here

#coding:utf-8

# 图片及视频检测结果保存路径
save_path = 'save_data'

# 使用的模型路径
model_path = 'models/best.pt'

names = {0: 'Trafic Light Signal' ,
  1: 'Stop Signal' ,
  2: 'Speedlimit Signal' ,
  3: 'Crosswalk Signal' ,
  4: 'Crosswalk',
  5: 'Pedestrian',
  6: 'Bus',
  7: 'Car',
  8: 'Truck'}
CH_names =  ['交通信号灯', '停止信号', '限速信号', '人行横道信号', '人行横道',
             '行人', '公交车', '汽车', '卡车', ]


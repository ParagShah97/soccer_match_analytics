from ultralytics import YOLO 

model = YOLO('best.pt')

results = model.predict('test_video.mp4',save=True)
# print(results[0])
# print('=====================================')
# for box in results[0].boxes:
#     print(box)
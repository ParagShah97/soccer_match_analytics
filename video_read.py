import cv2

def load_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        exit()
    video_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
    
    cap.release()
    cv2.destroyAllWindows()
    return video_frames

# def write_video(ouput_video_frames,output_video_path):
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
#     for frame in ouput_video_frames:
#         out.write(frame)
#     out.release()

def initialize_video_writer(output_path, frame_size, fps=24, codec='XVID'):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)

def write_frames(frames, writer):
    for frame in frames:
        writer.write(frame)
    writer.release()

def write_video(output_frames, output_path):
    if not output_frames:
        print("No frames provided to save.")
        return

    height, width = output_frames[0].shape[:2]
    writer = initialize_video_writer(output_path, (width, height))
    print("Run successfull")
    write_frames(output_frames, writer)



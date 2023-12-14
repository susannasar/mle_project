import streamlit as st
import cv2
import av
import io
import numpy as np
from detector import Detector

class App(object):
    def __init__(self):
        self.conf_threshold = 0.5
        self.classes = []
        self.detector = None
        self.images = []
        self.videos = []
        self.fps = 25

    def parse_labels(self):
        placeholder = st.sidebar.empty()
        with open('classes.txt', 'r') as uploaded_file:
            stringio = uploaded_file.read()
            self.classes = stringio.strip().split('\n')
            self.detector.set_classes(self.classes)
            placeholder.empty()

    def process_video(self, byte_array):
        container = av.open(byte_array)
        video_fps = round(float(container.streams.video[0].average_rate))
        frames = container.decode(video=0)

        output_binary_stream = io.BytesIO()
        output = av.open(output_binary_stream, 'w', format="mp4")
        stream = output.add_stream('libx264', self.fps)
        stream.pix_fmt = 'yuv420p'
        stream.width = 1280
        stream.height = 720

        self.fps = min(video_fps, self.fps)
        for i, frame in enumerate(frames):
            if (i % video_fps) % round(video_fps/ self.fps) != 0:
                continue
            img = frame.to_ndarray(format="bgr24")
            boxes, scores, classes = self.detector.predict(img)
            for k, box in enumerate(boxes):
               cv2.rectangle(img, box, (0, 0, 255), 2)
               cv2.putText(img, classes[k] + str(round(scores[k] * 100)) + '%', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            frame = av.VideoFrame.from_ndarray(img, format='bgr24')
            packet = stream.encode(frame)
            output.mux(packet)
        packet = stream.encode(None)
        output.mux(packet)
        
        output_binary_stream.seek(0)
        output.close()
        st.video(output_binary_stream)

    def process_image(self, bytes_array):
        img = cv2.imdecode(np.frombuffer(bytes_array, np.uint8), cv2.IMREAD_COLOR)
        boxes, scores, classes = self.detector.predict(img)
        for i, box in enumerate(boxes):
           cv2.rectangle(img, box, (0, 0, 255), 2)
           cv2.putText(img, classes[i] + str(round(scores[i] * 100)) + '%', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
        return img

    def create_sidebar(self):
        st.sidebar.title('Configuration')
        self.conf_threshold = st.sidebar.slider('Confidence threshold', 0.0, 1.0, 0.5)

        model_type = st.sidebar.selectbox(label='Model type', options=['', 'Yolov5s', 'Yolov5m'], placeholder="Choose an option")
        model_path = None
        if model_type == 'Yolov5s':
            model_path = 'models/yolov5s.onnx'
        elif model_type == 'Yolov5m':
            model_path = 'models/yolov5m.onnx'

        if model_path:
            if model_type not in st.session_state:
                st.session_state[model_type] = Detector()
                try:
                    with open(model_path, 'rb') as f:
                        st.session_state[model_type].load_net(f.read())
                        st.sidebar.success('Model is successfully loaded!')
                except Exception as e:
                    st.sidebar.error('Can not load model!', e)
            else:
                st.sidebar.success('Model is successfully loaded!')

            self.detector = st.session_state[model_type]
            self.detector.set_threshold(self.conf_threshold)

            self.parse_labels()
            with st.sidebar.expander('Classes'):
                st.text('\n'.join(self.classes))

        st.sidebar.title('Image')
        keep_ratio = st.sidebar.checkbox('Keep aspect ratio')
        if keep_ratio:
            self.detector.enable_keeping_aspect_ratio()
        st.sidebar.title('Video')
        self.fps = st.sidebar.number_input(label='fps', min_value=5, max_value=35, value=25)

    def process_uploaded_files(self, uploaded_files):
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            if file_name in self.images:
                continue
            self.images.append(file_name)
            if uploaded_file.type[:5] == 'image':
                img = self.process_image(uploaded_file.read())
                st.image(img, channels='BGR')
            elif uploaded_file.type[:5] == 'video':
                video = self.process_video(uploaded_file)
            else:
                st.error('Unsuppored file.')

    def create_main_container(self):
        st.title('Object detection')
        with st.form('detection form'):
            uploaded_files = st.file_uploader('Choose files', accept_multiple_files=True, type=['jpg', 'png', '.mp4'])
            submitted = st.form_submit_button("Process")
            if submitted:
                if len(uploaded_files) > 0:
                    if not self.detector:
                        st.error('Model is not loaded!')
                        return

                    placeholder = st.empty()
                    with st.spinner('Processing'):
                        self.process_uploaded_files(uploaded_files)
                    st.success('Done!')
                else:
                    st.error('No files are uploaded! Please upload files.')

    def run(self):
        self.create_sidebar()
        self.create_main_container()

if __name__ == '__main__':
    app = App()
    app.run()

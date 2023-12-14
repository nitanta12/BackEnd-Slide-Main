import os
import tempfile
from subprocess import call

__author__ = ['slideit']

## Sometimes ffmpeg is avconv
FFMPEG_NAME = 'ffmpeg'
stored_folder = 'output/'
image_loc= stored_folder + 'images/'
audio_loc = stored_folder + 'audio/'

output_path = "output.mp4"

def concat_audio_video(video_list_str, out_path):
    call([FFMPEG_NAME, '-y', '-f', 'mpegts', '-i', '{}'.format(video_list_str),
          '-c', 'copy', '-bsf:a', 'aac_adtstoasc', out_path])

def generate_video_from_(image_path, audio_path, temp_path, i):
    out_path_mp4 = os.path.join(temp_path, 'frame_{}.mp4'.format(i))
    out_path_ts = os.path.join(temp_path, 'frame_{}.ts'.format(i))
    call([FFMPEG_NAME, '-loop', '1', '-y', '-i', image_path, '-i', audio_path,
        '-c:v', 'libx264', '-tune', 'stillimage', '-c:a', 'aac',
        '-b:a', '192k', '-pix_fmt', 'yuv420p', '-shortest', '-vf', 'scale=w=trunc(iw/2)*2:h=trunc(ih/2)*2', out_path_mp4])
    call([FFMPEG_NAME, '-y', '-i', out_path_mp4, '-c', 'copy',
        '-bsf:v', 'h264_mp4toannexb', '-f', 'mpegts', out_path_ts])

def generate_video(num_of_slides):
    with tempfile.TemporaryDirectory() as temp_path:
        for i in range(num_of_slides):
            image_path = os.path.join(image_loc, 'frame_{}.jpg'.format(i))
            audio_path = os.path.join(audio_loc, 'frame_{}.wav'.format(i))
            generate_video_from_(image_path, audio_path, temp_path, i)

        video_list = [os.path.join(temp_path, 'frame_{}.ts'.format(i)) \
                        for i in range(num_of_slides)]
        video_list_str = 'concat:' + '|'.join(video_list)
        concat_audio_video (video_list_str, output_path)
    
    

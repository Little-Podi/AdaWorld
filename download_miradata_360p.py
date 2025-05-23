import argparse
import os
import pandas as pd
import requests
import tqdm
import youtube_dl
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--meta_csv", type=str, default="data/data_list/miradata_v1_330k.csv")
parser.add_argument("--raw_video_save_dir", type=str, default="miradata/raw_video")
parser.add_argument("--clip_video_save_dir", type=str, default="miradata/")
args = parser.parse_args()

df = pd.read_csv(args.meta_csv, encoding='utf-8')
print(f"Successfully loaded the csv file")

for i, row in tqdm.tqdm(df.iterrows()):
    if "youtube" in row["source"]:
        download_id = int(row["clip_id"])
        raw_video_download_path = os.path.join(args.raw_video_save_dir, str(download_id).zfill(12) + ".mp4")

        if not os.path.exists(raw_video_download_path):
            if not os.path.exists(os.path.dirname(raw_video_download_path)):
                os.makedirs(os.path.dirname(raw_video_download_path))

            # Download
            try:
                if "youtube" in row["source"]:
                    ret = os.system(f"yt-dlp -S vcodec:h264,res:360,acodec:aac -o '{raw_video_download_path}' -- {row['video_url'].split('watch?v=')[-1]}")
                else:
                    res = requests.get(row["video_url"], stream=True)
                    if os.path.exists(raw_video_download_path + ".tmp"):
                        os.remove(raw_video_download_path + ".tmp")
                    with open(raw_video_download_path + ".tmp", "wb") as f:
                        for chunk in res.iter_content(chunk_size=10240):
                            f.write(chunk)
                    os.rename(raw_video_download_path + ".tmp", raw_video_download_path)
            except Exception as error:
                print(error)

        # Cut
        try:
            clip_video_path = os.path.join(args.clip_video_save_dir, row["file_path"])
            if os.path.exists(clip_video_path):
                continue
            run_ss = eval(row["timestamp"])[0]
            run_t = str(datetime.strptime(eval(row["timestamp"])[1], "%H:%M:%S.%f") - datetime.strptime(
                eval(row["timestamp"])[0], "%H:%M:%S.%f"))
            run_command = f"ffmpeg -ss {run_ss} -t {run_t} -i {raw_video_download_path} -c copy -y {clip_video_path}"

            if not os.path.exists(os.path.dirname(clip_video_path)):
                os.makedirs(os.path.dirname(clip_video_path))

            os.system(run_command)
        except Exception as error:
            print(error)

print("Finish")
